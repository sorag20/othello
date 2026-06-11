<template>
  <div class="game-view">
    <header class="view-header">
      <h1 class="title">オセロ AI</h1>
      <router-link to="/history" class="nav-link">対局履歴 →</router-link>
    </header>

    <GameStatus
      :current-turn="gameStore.currentTurn"
      :stone-count="gameStore.stoneCount"
      :is-thinking="gameStore.isThinking"
      :is-game-over="gameStore.isGameOver"
      :passed="gameStore.passed"
    />

    <div class="board-area">
      <GameBoard
        :board="gameStore.board"
        :legal-moves="gameStore.legalMoves"
        :disabled="!gameStore.isPlayerTurn || gameStore.isThinking || gameStore.isGameOver"
        :current-turn="gameStore.currentTurn"
        @cell-click="handleCellClick"
      />
    </div>

    <ThinkingIndicator :visible="gameStore.isThinking" />

    <GameResult
      :result="gameStore.result"
      :stone-count="gameStore.stoneCount"
      :visible="gameStore.isGameOver && gameStore.result !== null"
      @new-game="handleNewGame"
    />

    <!-- Error toast -->
    <Transition name="error-fade">
      <div
        v-if="gameStore.error"
        class="error-toast"
        role="alert"
        aria-live="assertive"
      >
        <span class="error-message">{{ gameStore.error }}</span>
        <button class="error-dismiss" @click="gameStore.clearError()" aria-label="閉じる">×</button>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { onMounted, nextTick } from 'vue'
import { useGameStore } from '../stores/gameStore'
import GameBoard from '../components/GameBoard.vue'
import GameStatus from '../components/GameStatus.vue'
import ThinkingIndicator from '../components/ThinkingIndicator.vue'
import GameResult from '../components/GameResult.vue'

const gameStore = useGameStore()

/**
 * マウント時に自動的に新規ゲームを開始
 */
onMounted(() => {
  gameStore.startNewGame()
})

/**
 * プレイヤーのセルクリック処理
 * - 思考中インジケータ表示 & ボード無効化
 * - API呼び出し（レスポンスにはAIの手も含まれる）
 * - 思考中インジケータ非表示
 * - ボード更新は500ms以内に完了
 */
async function handleCellClick(row, col) {
  if (gameStore.isGameOver) return
  if (!gameStore.isPlayerTurn) return
  if (gameStore.isThinking) return

  gameStore.clearError()

  // 1. 石を楽観的に描画してから
  gameStore.applyOptimisticMove(row, col)
  await nextTick()

  // 2. AI 思考中オーバーレイを表示
  gameStore.setThinking(true)

  try {
    await gameStore.makeMove(row, col)
  } catch {
    // Error is already set in gameStore by makeMove
    // Auto-clear error after 3 seconds to allow retry
    setTimeout(() => {
      gameStore.clearError()
    }, 3000)
  } finally {
    gameStore.setThinking(false)
  }
}

/**
 * 新しいゲーム開始処理
 * GameResultモーダルからのイベントで呼び出される
 */
function handleNewGame() {
  gameStore.startNewGame()
}
</script>

<style scoped>
.game-view {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  position: relative;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.title {
  text-align: center;
  font-size: 1.6em;
  color: #333;
  margin: 0;
}

.nav-link {
  color: #2e7d32;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
}

.nav-link:hover {
  text-decoration: underline;
}

.board-area {
  display: flex;
  justify-content: center;
  margin: 16px 0;
}

/* Error toast */
.error-toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  background: #c62828;
  color: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 200;
  max-width: 90vw;
}

.error-message {
  font-size: 0.9em;
  font-weight: 500;
}

.error-dismiss {
  background: none;
  border: none;
  color: white;
  font-size: 1.2em;
  cursor: pointer;
  padding: 0 4px;
  opacity: 0.8;
  transition: opacity 0.2s;
}

.error-dismiss:hover {
  opacity: 1;
}

.error-fade-enter-active,
.error-fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.error-fade-enter-from,
.error-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(10px);
}

/* Responsive */
@media (max-width: 480px) {
  .game-view {
    padding: 12px;
  }

  .title {
    font-size: 1.3em;
  }

  .error-toast {
    bottom: 16px;
    padding: 10px 16px;
  }
}
</style>
