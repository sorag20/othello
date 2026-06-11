<template>
  <Transition name="result-fade">
    <div
      v-if="visible"
      class="game-result-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="result-title"
      @keydown.tab="trapFocus"
    >
      <div class="game-result-modal" ref="modalRef">
        <h2 id="result-title" class="result-title">対局終了</h2>

        <div class="result-outcome" aria-live="polite">
          <span v-if="result === 'black_win'" class="result-text win">
            🎉 あなたの勝ち！
          </span>
          <span v-else-if="result === 'white_win'" class="result-text lose">
            AIの勝ち
          </span>
          <span v-else-if="result === 'draw'" class="result-text draw">
            引き分け
          </span>
        </div>

        <div class="final-scores" aria-label="最終スコア">
          <div class="score-item">
            <span class="score-stone" aria-hidden="true">●</span>
            <span class="score-label">黒</span>
            <span class="score-value">{{ stoneCount.black }}</span>
          </div>
          <div class="score-separator" aria-hidden="true">—</div>
          <div class="score-item">
            <span class="score-stone score-stone-white" aria-hidden="true">○</span>
            <span class="score-label">白</span>
            <span class="score-value">{{ stoneCount.white }}</span>
          </div>
        </div>

        <button
          ref="newGameBtnRef"
          class="new-game-btn"
          @click="$emit('new-game')"
          aria-label="新しいゲームを始める"
        >
          新しいゲームを始める
        </button>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  result: {
    type: String,
    default: null,
    validator: (value) => value === null || ['black_win', 'white_win', 'draw'].includes(value)
  },
  stoneCount: {
    type: Object,
    required: true
  },
  visible: {
    type: Boolean,
    default: false
  }
})

defineEmits(['new-game'])

const modalRef = ref(null)
const newGameBtnRef = ref(null)

// Focus the new game button when modal becomes visible
watch(() => props.visible, async (newVal) => {
  if (newVal) {
    await nextTick()
    newGameBtnRef.value?.focus()
  }
})

// Focus trap within the modal
function trapFocus(event) {
  if (!modalRef.value) return
  const focusableElements = modalRef.value.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
  const firstElement = focusableElements[0]
  const lastElement = focusableElements[focusableElements.length - 1]

  if (event.shiftKey && document.activeElement === firstElement) {
    event.preventDefault()
    lastElement?.focus()
  } else if (!event.shiftKey && document.activeElement === lastElement) {
    event.preventDefault()
    firstElement?.focus()
  }
}
</script>

<style scoped>
.game-result-overlay {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
  z-index: 100;
  backdrop-filter: blur(2px);
}

.game-result-modal {
  background: white;
  border-radius: 16px;
  padding: 32px 40px;
  text-align: center;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  max-width: 360px;
  width: 90%;
  animation: modalPop 0.3s ease-out;
}

@keyframes modalPop {
  from {
    transform: scale(0.9);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.result-title {
  font-size: 1.1em;
  color: #666;
  margin: 0 0 16px;
  font-weight: 500;
}

.result-outcome {
  margin-bottom: 24px;
}

.result-text {
  font-size: 1.8em;
  font-weight: bold;
  display: block;
}

.result-text.win {
  color: #2e7d32;
}

.result-text.lose {
  color: #c62828;
}

.result-text.draw {
  color: #f57f17;
}

.final-scores {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-bottom: 28px;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 8px;
}

.score-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.score-stone {
  font-size: 1.2em;
  color: #1a1a1a;
}

.score-stone-white {
  color: #666;
}

.score-label {
  font-size: 0.85em;
  color: #666;
}

.score-value {
  font-size: 1.4em;
  font-weight: bold;
  color: #333;
}

.score-separator {
  color: #ccc;
  font-size: 1.2em;
}

.new-game-btn {
  padding: 12px 32px;
  font-size: 1em;
  font-weight: 600;
  cursor: pointer;
  border: none;
  border-radius: 8px;
  background: #2e7d32;
  color: white;
  transition: background 0.2s ease, transform 0.1s ease;
}

.new-game-btn:hover {
  background: #1b5e20;
}

.new-game-btn:active {
  transform: scale(0.97);
}

.new-game-btn:focus-visible {
  outline: 3px solid #66bb6a;
  outline-offset: 2px;
}

.result-fade-enter-active,
.result-fade-leave-active {
  transition: opacity 0.3s ease;
}

.result-fade-enter-from,
.result-fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 480px) {
  .game-result-modal {
    padding: 24px 20px;
    max-width: 300px;
  }

  .result-text {
    font-size: 1.5em;
  }

  .new-game-btn {
    padding: 10px 24px;
    font-size: 0.9em;
  }
}
</style>
