import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import BoardCell from './BoardCell.vue'

describe('BoardCell', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const defaultProps = {
    state: 0,
    row: 3,
    col: 4,
    isLegalMove: false,
    disabled: false
  }

  it('renders an empty cell with green background', () => {
    const wrapper = mount(BoardCell, { props: defaultProps })
    expect(wrapper.find('.board-cell').exists()).toBe(true)
    expect(wrapper.find('.stone').exists()).toBe(false)
    expect(wrapper.find('.legal-indicator').exists()).toBe(false)
  })

  it('renders a black stone when state is 1', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, state: 1 } })
    expect(wrapper.find('.stone.black').exists()).toBe(true)
    expect(wrapper.find('.stone.white').exists()).toBe(false)
  })

  it('renders a white stone when state is 2', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, state: 2 } })
    expect(wrapper.find('.stone.white').exists()).toBe(true)
    expect(wrapper.find('.stone.black').exists()).toBe(false)
  })

  it('shows legal move indicator on empty legal cell', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true } })
    expect(wrapper.find('.legal-indicator').exists()).toBe(true)
  })

  it('does not show legal indicator when cell has a stone', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, state: 1, isLegalMove: true } })
    expect(wrapper.find('.legal-indicator').exists()).toBe(false)
    expect(wrapper.find('.stone.black').exists()).toBe(true)
  })

  it('emits click event with row and col when legal move is clicked', async () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true } })
    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toHaveLength(1)
    expect(wrapper.emitted('click')[0]).toEqual([3, 4])
  })

  it('does not emit click when disabled', async () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true, disabled: true } })
    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toBeUndefined()
  })

  it('emits invalid-click for non-legal empty cell', async () => {
    const wrapper = mount(BoardCell, { props: defaultProps })
    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toBeUndefined()
    expect(wrapper.emitted('invalid-click')).toHaveLength(1)
    expect(wrapper.emitted('invalid-click')[0]).toEqual([3, 4])
  })

  it('shows error feedback on invalid click for at least 1 second', async () => {
    const wrapper = mount(BoardCell, { props: defaultProps })
    await wrapper.trigger('click')
    expect(wrapper.find('.board-cell').classes()).toContain('show-error')

    // Still showing after 900ms
    vi.advanceTimersByTime(900)
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.board-cell').classes()).toContain('show-error')

    // Gone after full timeout (1200ms)
    vi.advanceTimersByTime(400)
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.board-cell').classes()).not.toContain('show-error')
  })

  it('does not emit anything when disabled cell is clicked', async () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, disabled: true } })
    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toBeUndefined()
    expect(wrapper.emitted('invalid-click')).toBeUndefined()
  })

  it('has correct aria-label for empty cell', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, row: 0, col: 0 } })
    expect(wrapper.attributes('aria-label')).toBe('A1 空')
  })

  it('has correct aria-label for black stone', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, state: 1, row: 2, col: 3 } })
    expect(wrapper.attributes('aria-label')).toBe('D3 黒石')
  })

  it('has correct aria-label for legal move cell', () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true, row: 4, col: 5 } })
    expect(wrapper.attributes('aria-label')).toBe('F5 配置可能')
  })

  it('supports keyboard navigation with Enter key', async () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true } })
    await wrapper.trigger('keydown.enter')
    expect(wrapper.emitted('click')).toHaveLength(1)
  })

  it('supports keyboard navigation with Space key', async () => {
    const wrapper = mount(BoardCell, { props: { ...defaultProps, isLegalMove: true } })
    await wrapper.trigger('keydown.space')
    expect(wrapper.emitted('click')).toHaveLength(1)
  })
})
