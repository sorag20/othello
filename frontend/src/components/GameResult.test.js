import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import GameResult from './GameResult.vue'

describe('GameResult', () => {
  const defaultProps = {
    result: null,
    stoneCount: { black: 32, white: 32 },
    visible: false
  }

  it('is not visible when visible is false', () => {
    const wrapper = mount(GameResult, { props: defaultProps })
    expect(wrapper.find('.game-result-overlay').exists()).toBe(false)
  })

  it('is visible when visible is true and result is provided', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'draw', stoneCount: { black: 32, white: 32 }, visible: true }
    })
    expect(wrapper.find('.game-result-overlay').exists()).toBe(true)
  })

  it('displays "あなたの勝ち！" when result is black_win', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'black_win', stoneCount: { black: 40, white: 24 }, visible: true }
    })
    expect(wrapper.text()).toContain('あなたの勝ち！')
  })

  it('displays "AIの勝ち" when result is white_win', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'white_win', stoneCount: { black: 20, white: 44 }, visible: true }
    })
    expect(wrapper.text()).toContain('AIの勝ち')
  })

  it('displays "引き分け" when result is draw', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'draw', stoneCount: { black: 32, white: 32 }, visible: true }
    })
    expect(wrapper.text()).toContain('引き分け')
  })

  it('displays final stone counts', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'black_win', stoneCount: { black: 40, white: 24 }, visible: true }
    })
    expect(wrapper.text()).toContain('40')
    expect(wrapper.text()).toContain('24')
  })

  it('emits new-game event when button is clicked', async () => {
    const wrapper = mount(GameResult, {
      props: { result: 'black_win', stoneCount: { black: 40, white: 24 }, visible: true }
    })
    await wrapper.find('.new-game-btn').trigger('click')
    expect(wrapper.emitted('new-game')).toHaveLength(1)
  })

  it('has proper dialog accessibility attributes', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'draw', stoneCount: { black: 32, white: 32 }, visible: true }
    })
    const overlay = wrapper.find('.game-result-overlay')
    expect(overlay.attributes('role')).toBe('dialog')
    expect(overlay.attributes('aria-modal')).toBe('true')
    expect(overlay.attributes('aria-labelledby')).toBe('result-title')
  })

  it('contains "新しいゲームを始める" button with aria-label', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'black_win', stoneCount: { black: 40, white: 24 }, visible: true }
    })
    const btn = wrapper.find('.new-game-btn')
    expect(btn.exists()).toBe(true)
    expect(btn.text()).toBe('新しいゲームを始める')
    expect(btn.attributes('aria-label')).toBe('新しいゲームを始める')
  })

  it('shows stone count labels in Japanese', () => {
    const wrapper = mount(GameResult, {
      props: { result: 'black_win', stoneCount: { black: 40, white: 24 }, visible: true }
    })
    expect(wrapper.text()).toContain('黒')
    expect(wrapper.text()).toContain('白')
  })
})
