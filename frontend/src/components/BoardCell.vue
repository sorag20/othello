<template>
  <div
    class="board-cell"
    :class="{
      'is-legal-move': isLegalMove && state === 0,
      'has-stone': state !== 0,
      'is-disabled': disabled,
      'show-error': errorVisible
    }"
    role="gridcell"
    :aria-label="ariaLabel"
    :aria-disabled="disabled"
    tabindex="0"
    @click="handleClick"
    @keydown.enter="handleClick"
    @keydown.space.prevent="handleClick"
  >
    <div v-if="state === 1" class="stone black" aria-hidden="true"></div>
    <div v-else-if="state === 2" class="stone white" aria-hidden="true"></div>
    <div v-else-if="isLegalMove" class="legal-indicator" aria-hidden="true"></div>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount } from 'vue'

const props = defineProps({
  state: {
    type: Number,
    default: 0,
    validator: (v) => [0, 1, 2].includes(v)
  },
  row: {
    type: Number,
    required: true
  },
  col: {
    type: Number,
    required: true
  },
  isLegalMove: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['click', 'invalid-click'])

const errorVisible = ref(false)
let errorTimeout = null

const ariaLabel = computed(() => {
  const colLabel = String.fromCharCode(65 + props.col) // A-H
  const rowLabel = props.row + 1 // 1-8
  let stoneInfo = '空'
  if (props.state === 1) stoneInfo = '黒石'
  else if (props.state === 2) stoneInfo = '白石'
  else if (props.isLegalMove) stoneInfo = '配置可能'
  return `${colLabel}${rowLabel} ${stoneInfo}`
})

function handleClick() {
  if (props.disabled) return

  if (props.isLegalMove && props.state === 0) {
    emit('click', props.row, props.col)
  } else {
    // Invalid move - show error feedback for at least 1 second
    showErrorFeedback()
    emit('invalid-click', props.row, props.col)
  }
}

function showErrorFeedback() {
  if (errorTimeout) {
    clearTimeout(errorTimeout)
  }
  errorVisible.value = true
  errorTimeout = setTimeout(() => {
    errorVisible.value = false
    errorTimeout = null
  }, 1200) // Slightly over 1 second to ensure 1+ second visibility
}

onBeforeUnmount(() => {
  if (errorTimeout) {
    clearTimeout(errorTimeout)
  }
})
</script>

<style scoped>
.board-cell {
  aspect-ratio: 1;
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #1a6b1a;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  position: relative;
  transition: background-color 0.15s ease;
  background-color: #2e7d32;
}

.board-cell:hover:not(.is-disabled) {
  background-color: #388e3c;
}

.board-cell.is-legal-move:hover:not(.is-disabled) {
  background-color: #43a047;
}

.board-cell.is-disabled {
  cursor: not-allowed;
}

.board-cell.show-error {
  animation: error-flash 1.2s ease forwards;
  box-shadow: inset 0 0 8px rgba(231, 76, 60, 0.8);
}

@keyframes error-flash {
  0% {
    background-color: #c0392b;
    box-shadow: inset 0 0 12px rgba(231, 76, 60, 1);
  }
  30% {
    background-color: #e74c3c;
    box-shadow: inset 0 0 8px rgba(231, 76, 60, 0.8);
  }
  60% {
    background-color: #c0392b;
    box-shadow: inset 0 0 6px rgba(231, 76, 60, 0.5);
  }
  100% {
    background-color: #2e7d32;
    box-shadow: none;
  }
}

.stone {
  width: 75%;
  height: 75%;
  border-radius: 50%;
  transition: transform 0.3s ease, opacity 0.3s ease;
  animation: stone-place 0.3s ease;
}

@keyframes stone-place {
  0% {
    transform: scale(0);
    opacity: 0;
  }
  60% {
    transform: scale(1.1);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

.stone.black {
  background: radial-gradient(circle at 35% 35%, #555, #111);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
}

.stone.white {
  background: radial-gradient(circle at 35% 35%, #fff, #ddd);
  border: 1px solid #bbb;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

.legal-indicator {
  width: 30%;
  height: 30%;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.35);
  transition: transform 0.15s ease, background-color 0.15s ease;
}

.board-cell.is-legal-move:hover:not(.is-disabled) .legal-indicator {
  transform: scale(1.4);
  background-color: rgba(255, 255, 255, 0.55);
}
</style>
