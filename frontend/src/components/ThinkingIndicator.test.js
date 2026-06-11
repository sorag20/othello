import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ThinkingIndicator from './ThinkingIndicator.vue'

describe('ThinkingIndicator', () => {
  it('is visible when visible prop is true', () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: true } })
    expect(wrapper.find('.thinking-indicator').exists()).toBe(true)
  })

  it('is hidden when visible prop is false', () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: false } })
    expect(wrapper.find('.thinking-indicator').exists()).toBe(false)
  })

  it('displays "AIが考え中..." text', () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: true } })
    expect(wrapper.text()).toContain('AIが考え中...')
  })

  it('renders three animated dots', () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: true } })
    const dots = wrapper.findAll('.dot')
    expect(dots).toHaveLength(3)
  })

  it('has proper aria attributes for accessibility', () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: true } })
    const indicator = wrapper.find('.thinking-indicator')
    expect(indicator.attributes('role')).toBe('status')
    expect(indicator.attributes('aria-live')).toBe('polite')
    expect(indicator.attributes('aria-label')).toBe('AIが考え中...')
  })

  it('hides when visible changes from true to false', async () => {
    const wrapper = mount(ThinkingIndicator, { props: { visible: true } })
    expect(wrapper.find('.thinking-indicator').exists()).toBe(true)

    await wrapper.setProps({ visible: false })
    expect(wrapper.find('.thinking-indicator').exists()).toBe(false)
  })
})
