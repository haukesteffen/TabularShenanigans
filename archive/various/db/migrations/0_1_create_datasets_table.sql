CREATE TABLE IF NOT EXISTS datasets (
    dataset_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id   INTEGER NOT NULL,
    table_name       TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (competition_id) REFERENCES competitions(competition_id)
);