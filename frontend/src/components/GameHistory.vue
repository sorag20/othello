<template>
  <div class="game-history">
    <!-- Loading state -->
    <div v-if="isLoading" class="loading-state">
      <span class="loading-spinner"></span>
      <span>読み込み中...</span>
    </div>

    <!-- Game list -->
    <ul v-else-if="games.length > 0" class="history-list">
      <li
        v-for="game in games"
        :key="game.id"
        class="history-item"
        role="button"
        tabindex="0"
        @click="$emit('select-game', game.id)"
        @keydown.enter="$emit('select-game', game.id)"
      >
        <span class="result-icon">{{ resultIcon(game.result) }}</span>
        <span class="result-text">{{ resultText(game.result) }}</span>
        <span class="move-count">{{ game.move_count }}手</span>
        <span class="date">{{ formatDate(game.ended_at) }}</span>
      </li>
    </ul>

    <!-- Empty state -->
    <div v-else class="empty-state">
      <p>対局履歴がありません</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  games: {
    type: Array,
    default: () => []
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select-game'])

function resultIcon(result) {
  switch (result) {
    case 'black_win':
      return '⚫'
    case 'white_win':
      return '⚪'
    case 'draw':
      return '△'
    default:
      return '-'
  }
}

function resultText(result) {
  switch (result) {
    case 'black_win':
      return '勝ち'
    case 'white_win':
      return '負け'
    case 'draw':
      return '引き分け'
    default:
      return '-'
  }
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  const year = d.getFullYear()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  const hours = String(d.getHours()).padStart(2, '0')
  const minutes = String(d.getMinutes()).padStart(2, '0')
  return `${year}/${month}/${day} ${hours}:${minutes}`
}
</script>

<style scoped>
.game-history {
  width: 100%;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px 0;
  color: #666;
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #ddd;
  border-top-color: #2e7d32;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.history-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.history-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  transition: background-color 0.15s ease;
  border-radius: 4px;
}

.history-item:hover {
  background-color: #f5f5f5;
}

.history-item:focus {
  outline: 2px solid #2e7d32;
  outline-offset: -2px;
}

.result-icon {
  font-size: 1.2em;
}

.result-text {
  font-weight: 600;
  min-width: 60px;
}

.move-count {
  color: #666;
  font-size: 0.9em;
}

.date {
  margin-left: auto;
  color: #888;
  font-size: 0.85em;
}

.empty-state {
  text-align: center;
  padding: 48px 16px;
  color: #888;
}

.empty-state p {
  font-size: 1em;
}
</style>
