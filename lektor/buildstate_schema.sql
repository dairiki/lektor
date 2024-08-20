-- Schema for the sqlite build state database

CREATE TABLE IF NOT EXISTS artifacts (
    artifact TEXT,
    source TEXT,
    source_mtime INTEGER,
    source_size INTEGER,
    source_checksum TEXT,
    is_dir INTEGER,
    is_primary_source INTEGER,
    PRIMARY KEY (artifact, source)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS artifacts_source ON artifacts (
    source
);

CREATE TABLE IF NOT EXISTS artifact_config_hashes (
    artifact TEXT,
    config_hash TEXT,
    PRIMARY KEY (artifact)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS dirty_sources (
    source TEXT,
    PRIMARY KEY (source)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS source_info (
    path TEXT,
    alt TEXT,
    lang TEXT,
    type TEXT,
    source TEXT,
    title TEXT,
    PRIMARY KEY (path, alt, lang)
) WITHOUT ROWID;
