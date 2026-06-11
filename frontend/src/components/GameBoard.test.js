import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import GameBoard from './GameBoard.vue'
import BoardCell from './BoardCell.vue'

describe('GameBoard', () => {
  function createInitialBoard() {
    const board = Array.from({ length: 8 }, () => Array(8).fill(0))
    board[3][3] = 2
    board[3][4] = 1
    board[4][3] = 1
    board[4][4] = 2
    return board
  }

  const defaultProps = {
    board: createInitialBoard(),
    legalMoves: [[2, 3], [3, 2], [4, 5], [5, 4]],
    disabled: false,
    currentTurn: 'black'
  }

  it('renders 64 BoardCell components for 8×8 grid', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const cells = wrapper.findAllComponents(BoardCell)
    expect(cells).toHaveLength(64)
  })

  it('uses CSS Grid with 8 columns', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const board = wrapper.find('.game-board')
    expect(board.exists()).toBe(true)
    // The class has grid-template-columns: repeat(8, 1fr)
    expect(board.classes()).not.toContain('is-disabled')
  })

  it('displays column labels A through H', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const colLabels = wrapper.findAll('.column-label')
    expect(colLabels).toHaveLength(8)
    expect(colLabels[0].text()).toBe('A')
    expect(colLabels[7].text()).toBe('H')
  })

  it('displays row labels 1 through 8', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const rowLabels = wrapper.findAll('.row-label')
    expect(rowLabels).toHaveLength(8)
    expect(rowLabels[0].text()).toBe('1')
    expect(rowLabels[7].text()).toBe('8')
  })

  it('passes correct state to each BoardCell', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const cells = wrapper.findAllComponents(BoardCell)

    // Center cells: [3][3]=white, [3][4]=black, [4][3]=black, [4][4]=white
    const cell33 = cells[3 * 8 + 3] // row 3, col 3
    const cell34 = cells[3 * 8 + 4] // row 3, col 4
    const cell43 = cells[4 * 8 + 3] // row 4, col 3
    const cell44 = cells[4 * 8 + 4] // row 4, col 4

    expect(cell33.props('state')).toBe(2)
    expect(cell34.props('state')).toBe(1)
    expect(cell43.props('state')).toBe(1)
    expect(cell44.props('state')).toBe(2)
  })

  it('marks cells as legal moves correctly', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const cells = wrapper.findAllComponents(BoardCell)

    // [2,3] is a legal move
    const legalCell = cells[2 * 8 + 3]
    expect(legalCell.props('isLegalMove')).toBe(true)

    // [0,0] is not
    const nonLegalCell = cells[0]
    expect(nonLegalCell.props('isLegalMove')).toBe(false)
  })

  it('emits cell-click when a legal cell is clicked', async () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const cells = wrapper.findAllComponents(BoardCell)

    // Click on legal move cell [2,3]
    const legalCell = cells[2 * 8 + 3]
    await legalCell.trigger('click')

    expect(wrapper.emitted('cell-click')).toHaveLength(1)
    expect(wrapper.emitted('cell-click')[0]).toEqual([2, 3])
  })

  it('does not emit cell-click when disabled', async () => {
    const wrapper = mount(GameBoard, { props: { ...defaultProps, disabled: true } })
    const cells = wrapper.findAllComponents(BoardCell)

    const legalCell = cells[2 * 8 + 3]
    await legalCell.trigger('click')

    expect(wrapper.emitted('cell-click')).toBeUndefined()
  })

  it('applies is-disabled class when disabled prop is true', () => {
    const wrapper = mount(GameBoard, { props: { ...defaultProps, disabled: true } })
    expect(wrapper.find('.game-board').classes()).toContain('is-disabled')
  })

  it('shows disabled overlay when disabled', () => {
    const wrapper = mount(GameBoard, { props: { ...defaultProps, disabled: true } })
    expect(wrapper.find('.disabled-overlay').exists()).toBe(true)
    expect(wrapper.find('.overlay-text').text()).toBe('AI思考中...')
  })

  it('does not show disabled overlay when enabled', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    expect(wrapper.find('.disabled-overlay').exists()).toBe(false)
  })

  it('has grid role for accessibility', () => {
    const wrapper = mount(GameBoard, { props: defaultProps })
    const board = wrapper.find('[role="grid"]')
    expect(board.exists()).toBe(true)
  })

  it('passes disabled prop to all cells', () => {
    const wrapper = mount(GameBoard, { props: { ...defaultProps, disabled: true } })
    const cells = wrapper.findAllComponents(BoardCell)
    cells.forEach(cell => {
      expect(cell.props('disabled')).toBe(true)
    })
  })
})
