<template>
  <Transition name="thinking-fade">
    <div
      v-if="visible"
      class="thinking-indicator"
      role="status"
      aria-live="polite"
      aria-label="AIが考え中..."
    >
      <div class="dots" aria-hidden="true">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
      <span class="thinking-text">AIが考え中...</span>
    </div>
  </Transition>
</template>

<script setup>
defineProps({
  visible: {
    type: Boolean,
    required: true
  }
})
</script>

<style scoped>
.thinking-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 12px 0;
  padding: 10px 20px;
  background: #f3e5f5;
  border-radius: 20px;
  width: fit-content;
  margin-left: auto;
  margin-right: auto;
}

.dots {
  display: flex;
  gap: 4px;
  align-items: center;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #7b1fa2;
  animation: pulse 1.4s ease-in-out infinite;
}

.dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%, 80%, 100% {
    transform: scale(0.6);
    opacity: 0.4;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.thinking-text {
  font-size: 0.9em;
  color: #4a148c;
  font-weight: 500;
}

.thinking-fade-enter-active,
.thinking-fade-leave-active {
  transition: opacity 0.3s ease;
}

.thinking-fade-enter-from,
.thinking-fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 480px) {
  .thinking-indicator {
    padding: 8px 16px;
    gap: 8px;
  }

  .thinking-text {
    font-size: 0.8em;
  }

  .dot {
    width: 6px;
    height: 6px;
  }
}
</style>
