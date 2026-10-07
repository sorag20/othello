<template>
  <Transition name="setup-fade">
    <div v-if="visible" class="setup-overlay" role="dialog" aria-modal="true" aria-labelledby="setup-title">
      <div class="setup-card">
        <h2 id="setup-title" class="setup-title">対戦設定</h2>

        <div class="section">
          <label class="section-label">対戦相手</label>
          <div class="radio-group">
            <label class="radio-option" :class="{ active: localOpponentType === 'internal' }">
              <input type="radio" v-model="localOpponentType" value="internal" />
              <span class="radio-text">
                <span class="radio-name">オリジナル AI</span>
                <span class="radio-desc">MCTS + ニューラルネット</span>
              </span>
            </label>
            <label
              class="radio-option"
              :class="{ active: localOpponentType === 'edax', disabled: !edaxAvailable }"
            >
              <input
                type="radio"
                v-model="localOpponentType"
                value="edax"
                :disabled="!edaxAvailable"
              />
              <span class="radio-text">
                <span class="radio-name">
                  Edax
                  <span v-if="!edaxAvailable" class="unavailable-badge">利用不可</span>
                </span>
                <span class="radio-desc">世界トップクラスのオセロエンジン</span>
              </span>
            </label>
            <label class="radio-option" :class="{ active: localOpponentType === 'minimax' }">
              <input type="radio" v-model="localOpponentType" value="minimax" />
              <span class="radio-text">
                <span class="radio-name">Minimax AI</span>
                <span class="radio-desc">αβ探索 + 完全読みの探索型エンジン</span>
              </span>
            </label>
          </div>
        </div>

        <Transition name="level-slide">
          <div v-if="localOpponentType === 'edax'" class="section">
            <label class="section-label">強さレベル</label>
            <div class="level-display">
              <span class="level-label">{{ levelLabel }}</span>
              <span class="level-number">Lv.{{ localEdaxLevel }}</span>
            </div>
            <input
              type="range"
              v-model.number="localEdaxLevel"
              min="1"
              max="21"
              step="1"
              class="level-slider"
              :aria-label="`強さレベル ${localEdaxLevel}`"
            />
            <div class="level-ticks">
              <span>弱</span>
              <span>強</span>
            </div>
          </div>
        </Transition>

        <button class="start-btn" @click="handleStart">ゲーム開始</button>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  edaxAvailable: { type: Boolean, default: false },
})

const emit = defineEmits(['start'])

const localOpponentType = ref('internal')
const localEdaxLevel = ref(5)

// Reset to internal AI when edax becomes unavailable
watch(() => props.edaxAvailable, (available) => {
  if (!available) localOpponentType.value = 'internal'
})

const levelLabel = computed(() => {
  const lv = localEdaxLevel.value
  if (lv <= 3) return 'とても弱い'
  if (lv <= 6) return '弱い'
  if (lv <= 10) return '普通'
  if (lv <= 14) return '強い'
  if (lv <= 18) return 'とても強い'
  return 'エキスパート'
})

function handleStart() {
  emit('start', {
    opponentType: localOpponentType.value,
    edaxLevel: localOpponentType.value === 'edax' ? localEdaxLevel.value : null,
  })
}
</script>

<style scoped>
.setup-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 300;
  padding: 16px;
}

.setup-card {
  background: #fff;
  border-radius: 16px;
  padding: 32px 28px;
  width: 100%;
  max-width: 380px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}

.setup-title {
  margin: 0 0 24px;
  font-size: 1.3em;
  color: #1b5e20;
  text-align: center;
}

.section {
  margin-bottom: 20px;
}

.section-label {
  display: block;
  font-size: 0.82em;
  font-weight: 600;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 10px;
}

/* Radio options */
.radio-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.radio-option {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  user-select: none;
}

.radio-option.active {
  border-color: #2e7d32;
  background: #f1f8f1;
}

.radio-option.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.radio-option input[type='radio'] {
  accent-color: #2e7d32;
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

.radio-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.radio-name {
  font-weight: 600;
  font-size: 0.95em;
  color: #222;
  display: flex;
  align-items: center;
  gap: 8px;
}

.radio-desc {
  font-size: 0.78em;
  color: #777;
}

.unavailable-badge {
  font-size: 0.72em;
  background: #ffcdd2;
  color: #c62828;
  padding: 1px 6px;
  border-radius: 4px;
  font-weight: 500;
}

/* Level slider */
.level-display {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}

.level-label {
  font-size: 0.9em;
  font-weight: 600;
  color: #2e7d32;
}

.level-number {
  font-size: 0.82em;
  color: #888;
}

.level-slider {
  width: 100%;
  accent-color: #2e7d32;
  cursor: pointer;
}

.level-ticks {
  display: flex;
  justify-content: space-between;
  font-size: 0.72em;
  color: #aaa;
  margin-top: 4px;
}

/* Start button */
.start-btn {
  margin-top: 8px;
  width: 100%;
  padding: 14px;
  background: #2e7d32;
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 1em;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s, transform 0.1s;
}

.start-btn:hover {
  background: #1b5e20;
}

.start-btn:active {
  transform: scale(0.98);
}

/* Transitions */
.setup-fade-enter-active,
.setup-fade-leave-active {
  transition: opacity 0.2s ease;
}
.setup-fade-enter-from,
.setup-fade-leave-to {
  opacity: 0;
}

.level-slide-enter-active,
.level-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease, max-height 0.25s ease;
  overflow: hidden;
  max-height: 120px;
}
.level-slide-enter-from,
.level-slide-leave-to {
  opacity: 0;
  transform: translateY(-6px);
  max-height: 0;
}
</style>
