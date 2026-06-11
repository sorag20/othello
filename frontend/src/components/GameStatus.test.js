import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import GameStatus from './GameStatus.vue'

describe('GameStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const defaultProps = {
    currentTurn: 'black',
    stoneCount: { black: 2, white: 2 },
    isThinking: false,
    passed: false
  }

  it('displays stone counts for both players', () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    const counts = wrapper.findAll('.count')
    expect(counts[0].text()).toBe('2')
    expect(counts[1].text()).toBe('2')
  })

  it('displays updated stone counts', () => {
    const wrapper = mount(GameStatus, {
      props: { ...defaultProps, stoneCount: { black: 10, white: 5 } }
    })
    const counts = wrapper.findAll('.count')
    expect(counts[0].text()).toBe('10')
    expect(counts[1].text()).toBe('5')
  })

  it('highlights the active player when it is black turn', () => {
    const wrapper = mount(GameStatus, { props: { ...defaultProps, currentTurn: 'black' } })
    const playerInfos = wrapper.findAll('.player-info')
    expect(playerInfos[0].classes()).toContain('active')
    expect(playerInfos[1].classes()).not.toContain('active')
  })

  it('highlights the active player when it is white turn', () => {
    const wrapper = mount(GameStatus, { props: { ...defaultProps, currentTurn: 'white' } })
    const playerInfos = wrapper.findAll('.player-info')
    expect(playerInfos[0].classes()).not.toContain('active')
    expect(playerInfos[1].classes()).toContain('active')
  })

  it('does not highlight black player when isThinking is true', () => {
    const wrapper = mount(GameStatus, {
      props: { ...defaultProps, currentTurn: 'black', isThinking: true }
    })
    const playerInfos = wrapper.findAll('.player-info')
    expect(playerInfos[0].classes()).not.toContain('active')
  })

  it('shows "●プレイヤーの番" when black turn', () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    expect(wrapper.text()).toContain('●プレイヤーの番')
  })

  it('shows "○AIの番" when white turn', () => {
    const wrapper = mount(GameStatus, { props: { ...defaultProps, currentTurn: 'white' } })
    expect(wrapper.text()).toContain('○AIの番')
  })

  it('shows pass notification when passed prop becomes true', async () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    expect(wrapper.text()).not.toContain('相手にパスされました')

    await wrapper.setProps({ passed: true })
    expect(wrapper.text()).toContain('相手にパスされました')
  })

  it('hides pass notification after 2 seconds', async () => {
    const wrapper = mount(GameStatus, { props: { ...defaultProps, passed: true } })
    expect(wrapper.text()).toContain('相手にパスされました')

    vi.advanceTimersByTime(2000)
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.pass-notification').exists()).toBe(false)
  })

  it('has aria-live on turn message for accessibility', () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    const turnMessage = wrapper.find('.turn-message')
    expect(turnMessage.attributes('aria-live')).toBe('polite')
  })

  it('has role="status" on the root element', () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    expect(wrapper.find('.game-status').attributes('role')).toBe('status')
  })

  it('displays stone icons with correct symbols', () => {
    const wrapper = mount(GameStatus, { props: defaultProps })
    expect(wrapper.find('.stone-black').text()).toBe('●')
    expect(wrapper.find('.stone-white').text()).toBe('○')
  })
})
