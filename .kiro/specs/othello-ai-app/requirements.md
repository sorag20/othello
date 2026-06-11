# Requirements Document

## Introduction

オセロ（リバーシ）をAIと対戦できるWebアプリケーション。AIはAlphaZeroアーキテクチャに基づきPyTorchで実装する。バックエンドはPython Flask、フロントエンドはVue.js、データベースはMySQLを使用し、Docker上で動作する構成とする。

## Glossary

- **Game_Board**: 8×8のオセロ盤。各マスは空、黒、白のいずれかの状態を持つ
- **Player**: ゲームに参加するユーザー（人間プレイヤー）
- **AI_Engine**: AlphaZeroアーキテクチャに基づくPyTorch実装のAIエンジン
- **Game_Server**: Python Flaskで実装されたバックエンドサーバー。ゲームロジックとAPI提供を担当する
- **Frontend**: Vue.jsで実装されたフロントエンドアプリケーション
- **Game_Session**: 1回のオセロ対局を表すデータ単位
- **MCTS**: Monte Carlo Tree Search。AlphaZeroで使用される探索アルゴリズム
- **Neural_Network**: AlphaZeroのポリシーネットワークとバリューネットワークを統合したニューラルネットワーク
- **Database**: MySQLで実装されたデータ永続化層
- **Docker_Environment**: アプリケーション全体を動作させるDockerコンテナ環境

## Requirements

### Requirement 1: ゲームボードの表示と操作

**User Story:** プレイヤーとして、オセロ盤の状態を視覚的に確認し、石を配置できるようにしたい。

#### Acceptance Criteria

1. WHEN the Player opens the application, THE Frontend SHALL display an 8×8 Game_Board with the initial position of 4 stones (2 black, 2 white) in the center, with the board oriented so that row 1 is at the top and column A is at the left
2. WHEN the Player's turn begins, THE Frontend SHALL visually distinguish all legal move positions on the Game_Board by displaying a placement indicator on each valid cell
3. WHEN the Player places a stone on a valid cell, THE Frontend SHALL update the Game_Board to reflect the placed stone and all flipped stones within 500ms
4. IF the Player clicks a cell that is not a legal move, THEN THE Frontend SHALL not place a stone and SHALL display a visual feedback indicating the move is invalid for at least 1 second
5. WHILE a game is in progress, THE Frontend SHALL display the current stone count for both black and white as numeric values
6. WHILE a game is in progress, THE Frontend SHALL display the current turn holder (Player or AI_Engine) by showing the active player's name or label in a visually distinct state
7. WHILE it is the AI_Engine's turn, THE Frontend SHALL disable Player input on the Game_Board and not accept stone placement clicks
8. IF the current player has no legal moves available, THEN THE Frontend SHALL display a notification indicating the turn will be passed to the opponent

### Requirement 2: オセロのゲームルール実装

**User Story:** プレイヤーとして、正しいオセロのルールに基づいてゲームを進行したい。

#### Acceptance Criteria

1. THE Game_Server SHALL enforce standard Othello rules: a stone can only be placed on an empty cell where it flips at least one opponent stone in any of the 8 directions (horizontal, vertical, diagonal)
2. IF the Player attempts to place a stone on a cell that is outside the board, already occupied, or does not flip at least one opponent stone, THEN THE Game_Server SHALL reject the move, preserve the current Game_Board state, and return an error response indicating the reason for rejection
3. WHEN a stone is placed, THE Game_Server SHALL flip all opponent stones that are enclosed between the placed stone and existing same-color stones in all 8 directions (horizontal, vertical, diagonal) simultaneously in a single operation
4. WHEN a player has no valid moves, THE Game_Server SHALL automatically pass the turn to the opponent and notify the Frontend that the turn was passed
5. WHEN both players consecutively have no valid moves or the Game_Board is full (64 cells occupied), THE Game_Server SHALL end the game and determine the winner as the player with the higher stone count, or declare a draw if both players have equal stone counts (32 each)
6. WHEN a player's turn begins, THE Game_Server SHALL calculate and return all legal move positions for the current player as a list of board coordinates

### Requirement 3: AI対戦モード

**User Story:** プレイヤーとして、AIと対戦してオセロのスキルを向上させたい。

#### Acceptance Criteria

1. WHEN the Player starts a new game, THE Game_Server SHALL create a new Game_Session with the Player as black and the AI_Engine as white, and return the initial Game_Board state and Game_Session identifier to the Frontend
2. WHEN it is the AI_Engine's turn, THE AI_Engine SHALL select a move within 5 seconds
3. IF the AI_Engine fails to select a move within 5 seconds, THEN THE Game_Server SHALL terminate the AI_Engine's computation, forfeit the AI_Engine's turn by passing, and notify the Frontend with an error message indicating a timeout occurred
4. WHEN the AI_Engine selects a move, THE Game_Server SHALL apply the move to the Game_Board and notify the Frontend with the updated Game_Board state, the AI_Engine's move position, and the list of flipped stone positions within 1 second
5. WHILE the AI_Engine is computing a move, THE Frontend SHALL display a thinking indicator to the Player
6. WHEN a game ends, THE Frontend SHALL display the final result (win, lose, or draw) and the final stone count for both black and white, along with an option to start a new game
7. IF the Player disconnects during an active Game_Session, THEN THE Game_Server SHALL preserve the Game_Session state for 300 seconds, allowing the Player to reconnect and resume the game

### Requirement 4: AlphaZero AIエンジン

**User Story:** 開発者として、AlphaZeroアーキテクチャに基づく強力なAIを実装し、プレイヤーに挑戦的な対戦相手を提供したい。

#### Acceptance Criteria

1. THE AI_Engine SHALL implement a Neural_Network using PyTorch that combines a policy head (outputting move probabilities) and a value head (outputting position evaluation) sharing a common residual network backbone
2. THE Neural_Network SHALL accept an 8×8 board state as input and output a policy vector of size 64 (one probability per board cell, normalized via softmax) and a scalar value in the range [-1, 1] (via tanh activation)
3. THE AI_Engine SHALL implement MCTS that uses the Neural_Network for leaf node evaluation and uses the policy output to guide child node selection via the PUCT algorithm
4. WHEN the AI_Engine performs MCTS, THE AI_Engine SHALL execute a configurable number of simulations (range: 1 to 10000, default: 800) before selecting a move
5. WHEN the AI_Engine completes MCTS, THE AI_Engine SHALL select the move with the highest visit count from the MCTS root node among legal moves only
6. THE AI_Engine SHALL support loading pre-trained model weights from a file in PyTorch serialization format (.pt or .pth)
7. IF the model weights file is not found or cannot be loaded, THEN THE AI_Engine SHALL return an error indicating the model loading failure and SHALL NOT proceed with move computation
8. IF the Neural_Network outputs probabilities for illegal moves, THEN THE AI_Engine SHALL mask illegal move probabilities to zero and re-normalize the remaining legal move probabilities before MCTS node expansion

### Requirement 5: ゲームデータの永続化

**User Story:** プレイヤーとして、過去の対局履歴を確認できるようにしたい。

#### Acceptance Criteria

1. WHEN a Game_Session ends, THE Game_Server SHALL save the game record to the Database, including the complete move sequence (up to 60 moves), the result (black win, white win, or draw), the game start timestamp, and the game end timestamp
2. WHEN the Player requests game history, THE Game_Server SHALL retrieve and return up to 20 Game_Session records per request in descending order of game end timestamp, each containing the unique identifier, result, game end timestamp, and total move count
3. THE Database SHALL store each Game_Session with a UUID as unique identifier, the Player's identifier, the move sequence encoded as an ordered list of positions played, the result, the game start timestamp, and the game end timestamp
4. WHEN the Player selects a past game from history, THE Frontend SHALL display the Game_Board and allow the Player to step through the move sequence one move at a time (forward and backward), showing the board state after each move
5. IF the Game_Server fails to save the game record to the Database when a Game_Session ends, THEN THE Game_Server SHALL retry the save operation up to 3 times, and if all retries fail, SHALL return an error message indicating that the game record could not be saved
6. IF the Player requests game history and no Game_Session records exist, THEN THE Game_Server SHALL return an empty list with a total count of 0

### Requirement 6: API設計

**User Story:** 開発者として、フロントエンドとバックエンド間の通信を明確に定義したい。

#### Acceptance Criteria

1. WHEN the Frontend sends a POST request to /api/games, THE Game_Server SHALL create a new Game_Session and return a JSON response containing the game ID, the initial 8×8 board state, the current turn, and the list of legal moves
2. WHEN the Frontend sends a POST request to /api/games/{id}/moves with a move position (row and column), THE Game_Server SHALL apply the move and return a JSON response containing the updated board state, flipped stones, the next turn, the list of legal moves, and the game status (in_progress, finished)
3. WHEN the Frontend sends a GET request to /api/games/{id}, THE Game_Server SHALL return a JSON response containing the board state, current turn, stone counts for both players, legal moves, and game status
4. WHEN the Frontend sends a GET request to /api/games, THE Game_Server SHALL return a JSON response containing a list of past Game_Session summaries (game ID, result, timestamps) with a maximum of 50 records per response, ordered by most recent first
5. THE Game_Server SHALL return HTTP status code 200 for successful retrieval requests, 201 for successful resource creation, 400 for invalid request parameters, 404 for non-existent game IDs, and 500 for internal server errors
6. IF the Frontend sends a request with a game ID that does not exist, THEN THE Game_Server SHALL return HTTP status code 404 with a JSON error response containing an error code and a message indicating the requested game was not found
7. IF the Frontend sends a move request with an invalid or missing position parameter, THEN THE Game_Server SHALL return HTTP status code 400 with a JSON error response containing an error code and a message indicating the validation failure reason
8. IF the Game_Server encounters an internal error during request processing, THEN THE Game_Server SHALL return HTTP status code 500 with a JSON error response containing an error code and a message indicating that an internal error occurred, without exposing internal system details

### Requirement 7: Docker環境構成

**User Story:** 開発者として、アプリケーション全体をDocker上で簡単に起動できるようにしたい。

#### Acceptance Criteria

1. THE Docker_Environment SHALL define a docker-compose.yml that orchestrates all services (Frontend, Game_Server, Database)
2. WHEN a developer runs `docker-compose up`, THE Docker_Environment SHALL start all services in dependency order (Database → Game_Server → Frontend) and make the Frontend accessible on the host machine at port 8080 within 120 seconds
3. THE Docker_Environment SHALL include a dedicated container for the Frontend (Vue.js) that serves the application and proxies API requests to the Game_Server
4. THE Docker_Environment SHALL include a dedicated container for the Game_Server (Python Flask with AI_Engine) that exposes the API on an internal port accessible to other containers
5. THE Docker_Environment SHALL include a dedicated container for the Database (MySQL) that initializes the required schema on first startup
6. THE Docker_Environment SHALL persist Database data using a named Docker volume so that data survives container restarts
7. THE Docker_Environment SHALL configure a shared Docker network enabling the Frontend to communicate with the Game_Server and the Game_Server to communicate with the Database
8. THE Docker_Environment SHALL define environment variables for service connection settings (database host, port, credentials, API base URL) in the docker-compose.yml
9. IF a dependent service (Database or Game_Server) is not ready, THEN THE Docker_Environment SHALL delay the startup of services that depend on it using health checks or wait mechanisms
10. IF a container fails to start or crashes, THEN THE Docker_Environment SHALL configure a restart policy (on-failure, maximum 3 retries) to attempt automatic recovery

### Requirement 8: Neural Networkの入出力フォーマット

**User Story:** 開発者として、Neural Networkの入出力を明確に定義し、正しく動作することを保証したい。

#### Acceptance Criteria

1. THE Neural_Network SHALL encode the board state as a float32 tensor with dimensions 3 × 8 × 8, where channel 0 represents the current player's stones (1.0 for occupied, 0.0 otherwise), channel 1 represents the opponent's stones (1.0 for occupied, 0.0 otherwise), and channel 2 represents the current player's turn (all 1.0 if black's turn, all 0.0 if white's turn)
2. THE Neural_Network SHALL output a policy vector of size 64 (one probability per board cell) normalized via softmax, where positions corresponding to illegal moves are masked to 0 before normalization so that only legal move positions have non-zero probability
3. THE Neural_Network SHALL output a value scalar in the range [-1, 1] via tanh activation, where 1 represents a winning position for the current player and -1 represents a losing position for the current player
4. THE Neural_Network SHALL guarantee that encoding a valid board state and then decoding the resulting tensor produces a board state where every cell has the identical state (empty, black, or white) as the original board state
5. THE Neural_Network SHALL guarantee that for any policy output, the sum of all 64 elements in the policy vector equals 1.0 within a floating point tolerance of 1e-6
6. IF the Neural_Network receives a board state where no legal moves exist for the current player, THEN THE Neural_Network SHALL output a uniform policy vector (1/64 for each cell) and a value scalar representing the position evaluation

### Requirement 9: セルフプレイによる学習

**User Story:** 開発者として、AIを自己対戦によって学習させ、モデルの強化を行いたい。

#### Acceptance Criteria

1. THE AI_Engine SHALL support a self-play mode where two instances of the AI_Engine using the same model weights play against each other to generate training data
2. WHEN self-play is executed, THE AI_Engine SHALL record each training sample consisting of the board state (channels × 8 × 8 tensor), the MCTS policy vector (size 64, normalized to sum 1.0), and the game result from the current player's perspective (1 for win, -1 for loss, 0 for draw)
3. WHEN self-play data collection is complete, THE AI_Engine SHALL train the Neural_Network using the collected self-play data to minimize the combined loss of policy cross-entropy and value mean-squared-error
4. WHEN training is complete, THE AI_Engine SHALL save the updated model weights to a file in PyTorch format (.pt)
5. THE AI_Engine SHALL support configurable self-play parameters with the following defaults and valid ranges: number of games (default: 100, range: 1–10,000), MCTS simulations per move (default: 800, range: 1–10,000), learning rate (default: 0.001, range: 0.0001–0.1), training batch size (default: 64, range: 1–1,024), and training epochs (default: 10, range: 1–1,000)
6. IF self-play is interrupted before all games are completed, THEN THE AI_Engine SHALL retain any training samples already collected from fully completed games and discard samples from the incomplete game
7. WHEN training completes, THE AI_Engine SHALL output a training summary including the number of games played, total training samples generated, and the final combined loss value
