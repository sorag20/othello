<template>
  <div class="game-board-wrapper">
    <!-- Column labels (A-H) -->
    <div class="column-labels">
      <div class="label-spacer"></div>
      <div
        v-for="col in 8"
        :key="'col-' + col"
        class="column-label"
      >
        {{ String.fromCharCode(64 + col) }}
      </div>
    </div>

    <div class="board-with-rows">
      <!-- Row labels -->
      <div class="row-labels">
        <div
          v-for="row in 8"
          :key="'row-' + row"
          class="row-label"
        >
          {{ row }}
        </div>
      </div>

      <!-- Board grid -->
      <div class="board-container">
        <div
          class="game-board"
          :class="{ 'is-disabled': disabled }"
          role="grid"
          :aria-label="'オセロ盤 - ' + turnLabel"
        >
          <BoardCell
            v-for="(cell, index) in flatBoard"
            :key="'cell-' + index"
            :state="cell"
            :row="Math.floor(index / 8)"
            :col="index % 8"
            :is-legal-move="isLegalMove(Math.floor(index / 8), index % 8)"
            :disabled="disabled"
            @click="handleCellClick"
            @invalid-click="handleInvalidClick"
          />
        </div>

        <!-- Disabled overlay for AI turn -->
        <div v-if="disabled" class="disabled-overlay">
          <span class="overlay-text">AI思考中...</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import BoardCell from './BoardCell.vue'

const props = defineProps({
  board: {
    type: Array,
    required: true,
    default: () => Array.from({ length: 8 }, () => Array(8).fill(0))
  },
  legalMoves: {
    type: Array,
    default: () => []
  },
  disabled: {
    type: Boolean,
    default: false
  },
  currentTurn: {
    type: String,
    default: 'black'
  }
})

const emit = defineEmits(['cell-click'])

const turnLabel = computed(() => {
  return props.currentTurn === 'black' ? '黒の手番' : '白の手番（AI思考中）'
})

// Flatten 8×8 array for CSS Grid iteration
const flatBoard = computed(() => {
  if (!props.board || props.board.length === 0) {
    return Array(64).fill(0)
  }
  return props.board.flat()
})

function isLegalMove(row, col) {
  return props.legalMoves.some(([r, c]) => r === row && c === col)
}

function handleCellClick(row, col) {
  if (props.disabled) return
  emit('cell-click', row, col)
}

function handleInvalidClick(row, col) {
  // Error feedback is handled by BoardCell component itself
}
</script>

<style scoped>
.game-board-wrapper {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  user-select: none;
}

.column-labels {
  display: flex;
  align-items: center;
}

.label-spacer {
  width: 28px;
  height: 28px;
}

.column-label {
  width: calc((min(70vw, 480px)) / 8);
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  color: #e0e0e0;
}

.board-with-rows {
  display: flex;
  align-items: flex-start;
}

.row-labels {
  display: flex;
  flex-direction: column;
}

.row-label {
  width: 28px;
  height: calc((min(70vw, 480px)) / 8);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  color: #e0e0e0;
}

.board-container {
  position: relative;
}

.game-board {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
  width: min(70vw, 480px);
  height: min(70vw, 480px);
  border: 3px solid #1b5e20;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  background-color: #2e7d32;
  transition: opacity 0.3s ease;
}

.game-board.is-disabled {
  opacity: 0.7;
}

.disabled-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.15);
  border-radius: 4px;
  pointer-events: none;
}

.overlay-text {
  background-color: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 500;
}
</style>
