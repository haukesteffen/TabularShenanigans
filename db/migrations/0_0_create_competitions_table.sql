CREATE TABLE IF NOT EXISTS competitions (
    competition_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);