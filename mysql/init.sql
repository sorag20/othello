-- MySQL initialization script for Othello AI application
-- Creates the database and game_sessions table with indexes.

CREATE DATABASE IF NOT EXISTS othello
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE othello;

CREATE TABLE IF NOT EXISTS game_sessions (
    id CHAR(36) PRIMARY KEY,
    player_id VARCHAR(255),
    result ENUM('black_win', 'white_win', 'draw') NULL,
    move_sequence JSON,
    status ENUM('in_progress', 'finished') NOT NULL DEFAULT 'in_progress',
    board_state JSON NOT NULL,
    current_turn ENUM('black', 'white') NOT NULL DEFAULT 'black',
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_player_ended (player_id, ended_at DESC),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
