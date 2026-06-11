<template>
  <div class="history-view">
    <header class="view-header">
      <h1 class="title">対局履歴</h1>
      <router-link to="/" class="nav-link">← 対局画面へ</router-link>
    </header>

    <!-- Replay mode -->
    <div v-if="replayData" class="replay-section">
      <button class="back-button" @click="closeReplay">← 履歴一覧に戻る</button>

      <div class="replay-board-area">
        <GameBoard
          :board="currentBoard"
          :legal-moves="[]"
          :disabled="true"
          current-turn="black"
        />
      </div>

      <div class="replay-controls">
        <button
          class="control-btn"
          :disabled="currentMoveIndex === 0"
          @click="goToFirst"
          aria-label="最初の手へ"
        >|←</button>
        <button
          class="control-btn"
          :disabled="currentMoveIndex === 0"
          @click="prevMove"
          aria-label="前の手へ"
        >←</button>
        <span class="move-indicator">
          {{ currentMoveIndex }} / {{ replayData.total_moves }}
        </span>
        <button
          class="control-btn"
          :disabled="currentMoveIndex >= replayData.total_moves"
          @click="nextMove"
          aria-label="次の手へ"
        >→</button>
        <button
          class="control-btn"
          :disabled="currentMoveIndex >= replayData.total_moves"
          @click="goToLast"
          aria-label="最後の手へ"
        >→|</button>
      </div>

      <div class="replay-info">
        <span class="replay-result">結果: {{ formatResult(replayData.result) }}</span>
      </div>
    </div>

    <!-- History list mode -->
    <div v-else>
      <GameHistory
        :games="games"
        :is-loading="isLoading"
        @select-game="handleSelectGame"
      />

      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { gameApi } from '../api/gameApi.js'
import GameHistory from '../components/GameHistory.vue'
import GameBoard from '../components/GameBoard.vue'

const games = ref([])
const isLoading = ref(false)
const error = ref(null)

// Replay state
const replayData = ref(null)
const currentMoveIndex = ref(0)

const currentBoard = computed(() => {
  if (!replayData.value || !replayData.value.board_states) return []
  return replayData.value.board_states[currentMoveIndex.value] || []
})

onMounted(async () => {
  await fetchHistory()
})

async function fetchHistory() {
  isLoading.value = true
  error.value = null
  try {
    const data = await gameApi.getHistory()
    games.value = data.games || []
  } catch (e) {
    error.value = e.message || '履歴の取得に失敗しました'
  } finally {
    isLoading.value = false
  }
}

async function handleSelectGame(gameId) {
  isLoading.value = true
  error.value = null
  try {
    const data = await gameApi.getReplay(gameId)
    replayData.value = data
    currentMoveIndex.value = 0
  } catch (e) {
    error.value = e.message || 'リプレイデータの取得に失敗しました'
  } finally {
    isLoading.value = false
  }
}

function closeReplay() {
  replayData.value = null
  currentMoveIndex.value = 0
}

function nextMove() {
  if (currentMoveIndex.value < replayData.value.total_moves) {
    currentMoveIndex.value++
  }
}

function prevMove() {
  if (currentMoveIndex.value > 0) {
    currentMoveIndex.value--
  }
}

function goToFirst() {
  currentMoveIndex.value = 0
}

function goToLast() {
  currentMoveIndex.value = replayData.value.total_moves
}

function formatResult(result) {
  switch (result) {
    case 'black_win':
      return '⚫ 黒の勝ち'
    case 'white_win':
      return '⚪ 白の勝ち'
    case 'draw':
      return '△ 引き分け'
    default:
      return '-'
  }
}
</script>

<style scoped>
.history-view {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.title {
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

.back-button {
  background: none;
  border: none;
  color: #2e7d32;
  font-size: 0.9em;
  font-weight: 500;
  cursor: pointer;
  padding: 4px 0;
  margin-bottom: 16px;
}

.back-button:hover {
  text-decoration: underline;
}

.replay-section {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.replay-board-area {
  display: flex;
  justify-content: center;
  margin: 16px 0;
}

.replay-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 16px 0;
}

.control-btn {
  width: 44px;
  height: 44px;
  border: 1px solid #ccc;
  border-radius: 6px;
  background: #fff;
  font-size: 1em;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

.control-btn:hover:not(:disabled) {
  background-color: #f0f0f0;
  border-color: #2e7d32;
}

.control-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.move-indicator {
  font-size: 1em;
  font-weight: 500;
  color: #333;
  min-width: 80px;
  text-align: center;
}

.replay-info {
  margin-top: 12px;
  font-size: 0.95em;
  color: #555;
}

.replay-result {
  font-weight: 500;
}

.error-message {
  margin-top: 16px;
  padding: 12px;
  background-color: #ffebee;
  color: #c62828;
  border-radius: 4px;
  font-size: 0.9em;
}
</style>
