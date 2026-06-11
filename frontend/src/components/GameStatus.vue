<template>
  <div class="game-status" role="status">
    <div class="stone-counts">
      <div
        class="player-info"
        :class="{ active: currentTurn === 'black' && !isThinking }"
        aria-label="黒の石数"
      >
        <span class="stone stone-black" aria-hidden="true">●</span>
        <span class="player-label">プレイヤー</span>
        <span class="count">{{ stoneCount.black }}</span>
      </div>
      <div class="vs-divider" aria-hidden="true">vs</div>
      <div
        class="player-info"
        :class="{ active: currentTurn === 'white' }"
        aria-label="白の石数"
      >
        <span class="stone stone-white" aria-hidden="true">○</span>
        <span class="player-label">AI</span>
        <span class="count">{{ stoneCount.white }}</span>
      </div>
    </div>

    <div class="turn-message" aria-live="polite" aria-atomic="true">
      <span v-if="currentTurn === 'black'" class="turn-text player-turn">
        ●プレイヤーの番
      </span>
      <span v-else class="turn-text ai-turn">
        ○AIの番
      </span>
    </div>

    <Transition name="pass-fade">
      <div
        v-if="showPassNotification"
        class="pass-notification"
        role="alert"
        aria-live="assertive"
      >
        相手にパスされました
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  currentTurn: {
    type: String,
    required: true,
    validator: (value) => ['black', 'white'].includes(value)
  },
  stoneCount: {
    type: Object,
    required: true
  },
  isThinking: {
    type: Boolean,
    default: false
  },
  passed: {
    type: Boolean,
    default: false
  }
})

const showPassNotification = ref(false)
let passTimer = null

watch(() => props.passed, (newVal) => {
  if (newVal) {
    showPassNotification.value = true
    if (passTimer) clearTimeout(passTimer)
    passTimer = setTimeout(() => {
      showPassNotification.value = false
    }, 2000)
  }
}, { immediate: true })
</script>

<style scoped>
.game-status {
  margin: 16px 0;
  text-align: center;
}

.stone-counts {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.player-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  border: 2px solid transparent;
  transition: all 0.3s ease;
  background: #f5f5f5;
}

.player-info.active {
  border-color: #4caf50;
  background: #e8f5e9;
  box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
  font-weight: bold;
}

.stone {
  font-size: 1.4em;
  line-height: 1;
}

.stone-black {
  color: #1a1a1a;
}

.stone-white {
  color: #666;
}

.player-label {
  font-size: 0.85em;
  color: #666;
  font-weight: 500;
}

.player-info.active .player-label {
  color: #2e7d32;
  font-weight: bold;
}

.count {
  font-size: 1.3em;
  font-weight: bold;
  color: #333;
  min-width: 24px;
  text-align: center;
}

.vs-divider {
  font-size: 0.8em;
  color: #999;
  font-weight: 500;
  text-transform: uppercase;
}

.turn-message {
  margin-top: 12px;
  min-height: 28px;
}

.turn-text {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.9em;
  font-weight: bold;
}

.player-turn {
  background: #e3f2fd;
  color: #1565c0;
}

.ai-turn {
  background: #fff3e0;
  color: #e65100;
}

.pass-notification {
  margin-top: 12px;
  padding: 8px 16px;
  background: #fff8e1;
  border: 1px solid #ffcc02;
  border-radius: 8px;
  color: #6d4c00;
  font-size: 0.9em;
  font-weight: 500;
}

.pass-fade-enter-active,
.pass-fade-leave-active {
  transition: opacity 0.4s ease;
}

.pass-fade-enter-from,
.pass-fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 480px) {
  .stone-counts {
    gap: 8px;
  }

  .player-info {
    padding: 6px 10px;
    gap: 4px;
  }

  .stone {
    font-size: 1.2em;
  }

  .count {
    font-size: 1.1em;
  }
}
</style>
