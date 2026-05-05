//! SQLite transcript cache with TTL support.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

pub fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("trans")
}

pub fn cache_db_path() -> PathBuf {
    cache_dir().join("transcripts.db")
}

fn init_db(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let conn = Connection::open(db_path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS transcripts (
            video_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            transcript TEXT,
            format TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    Ok(())
}

pub struct CacheManager {
    db_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub count: i64,
    pub size_mb: f64,
    pub oldest: Option<String>,
    pub newest: Option<String>,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            db_path: cache_db_path(),
        }
    }

    pub fn with_path(path: PathBuf) -> Self {
        Self { db_path: path }
    }

    /// Return (transcript, title) if cached and within TTL, else None.
    pub fn get(&self, video_id: &str, fmt: &str, ttl_days: i64) -> Result<Option<(String, String)>> {
        if !self.db_path.exists() {
            return Ok(None);
        }
        let conn = Connection::open(&self.db_path)?;
        let cutoff = format!("-{} days", ttl_days);
        let mut stmt = conn.prepare(
            "SELECT transcript, title FROM transcripts
             WHERE video_id = ?1 AND format = ?2
             AND created_at > datetime('now', ?3)",
        )?;
        let mut rows = stmt.query(params![video_id, fmt, cutoff])?;
        if let Some(row) = rows.next()? {
            let transcript: String = row.get(0)?;
            let title: String = row.get(1)?;
            return Ok(Some((transcript, title)));
        }
        Ok(None)
    }

    /// Store a transcript in the cache.
    pub fn put(
        &self,
        video_id: &str,
        url: &str,
        title: &str,
        transcript: &str,
        fmt: &str,
        model: Option<&str>,
    ) -> Result<()> {
        init_db(&self.db_path).context("init cache db")?;
        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT OR REPLACE INTO transcripts
             (video_id, url, title, transcript, format, model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![video_id, url, title, transcript, fmt, model],
        )?;
        Ok(())
    }

    /// Delete all cached entries. Returns rows deleted.
    pub fn clear(&self) -> Result<i64> {
        if !self.db_path.exists() {
            return Ok(0);
        }
        let conn = Connection::open(&self.db_path)?;
        let n = conn.execute("DELETE FROM transcripts", [])?;
        Ok(n as i64)
    }

    pub fn stats(&self) -> Result<CacheStats> {
        if !self.db_path.exists() {
            return Ok(CacheStats {
                count: 0,
                size_mb: 0.0,
                oldest: None,
                newest: None,
            });
        }
        let conn = Connection::open(&self.db_path)?;
        let mut stmt = conn
            .prepare("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM transcripts")?;
        let mut rows = stmt.query([])?;
        let (count, oldest, newest) = if let Some(row) = rows.next()? {
            let c: i64 = row.get::<_, Option<i64>>(0)?.unwrap_or(0);
            let o: Option<String> = row.get(1)?;
            let n: Option<String> = row.get(2)?;
            (c, o, n)
        } else {
            (0, None, None)
        };
        let size_mb = self
            .db_path
            .metadata()
            .map(|m| (m.len() as f64) / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        Ok(CacheStats {
            count,
            size_mb: (size_mb * 100.0).round() / 100.0,
            oldest,
            newest,
        })
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}
