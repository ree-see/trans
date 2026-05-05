//! Cache round-trip and stats tests using a temp DB path.

use tempfile::tempdir;
use trans::cache::CacheManager;

#[test]
fn put_then_get_round_trip() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("transcripts.db");
    let cache = CacheManager::with_path(db);

    cache
        .put("yt_abc12345678", "https://x", "hello", "body", "txt", Some("base"))
        .unwrap();
    let got = cache.get("yt_abc12345678", "txt", 30).unwrap();
    assert_eq!(got.unwrap(), ("body".to_string(), "hello".to_string()));
}

#[test]
fn get_returns_none_for_missing_db() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("nonexistent.db");
    let cache = CacheManager::with_path(db);
    assert!(cache.get("yt_abc", "txt", 30).unwrap().is_none());
}

#[test]
fn clear_returns_count_then_empty() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("transcripts.db");
    let cache = CacheManager::with_path(db);
    cache.put("a", "u1", "t1", "b1", "txt", None).unwrap();
    cache.put("b", "u2", "t2", "b2", "txt", None).unwrap();
    let n = cache.clear().unwrap();
    assert_eq!(n, 2);
    assert!(cache.get("a", "txt", 30).unwrap().is_none());
}

#[test]
fn stats_empty_db() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("missing.db");
    let cache = CacheManager::with_path(db);
    let s = cache.stats().unwrap();
    assert_eq!(s.count, 0);
    assert_eq!(s.size_mb, 0.0);
    assert!(s.oldest.is_none());
    assert!(s.newest.is_none());
}

#[test]
fn stats_populated() {
    let tmp = tempdir().unwrap();
    let db = tmp.path().join("transcripts.db");
    let cache = CacheManager::with_path(db);
    cache.put("a", "u", "t", "b", "txt", None).unwrap();
    let s = cache.stats().unwrap();
    assert_eq!(s.count, 1);
    assert!(s.oldest.is_some());
    assert!(s.newest.is_some());
}
