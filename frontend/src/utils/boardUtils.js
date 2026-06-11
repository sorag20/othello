/**
 * ボード操作ユーティリティ
 */

/**
 * 初期盤面を生成
 * @returns {number[][]} 8×8の盤面配列 (0=empty, 1=black, 2=white)
 */
export function createInitialBoard() {
  const board = Array.from({ length: 8 }, () => Array(8).fill(0))
  board[3][3] = 2 // white
  board[3][4] = 1 // black
  board[4][3] = 1 // black
  board[4][4] = 2 // white
  return board
}

/**
 * 指定位置が合法手かどうか判定
 * @param {[number, number][]} legalMoves - 合法手リスト
 * @param {number} row
 * @param {number} col
 * @returns {boolean}
 */
export function isLegalMove(legalMoves, row, col) {
  return legalMoves.some(([r, c]) => r === row && c === col)
}

const DIRECTIONS = [
  [-1, -1], [-1, 0], [-1, 1],
  [ 0, -1],          [ 0, 1],
  [ 1, -1], [ 1, 0], [ 1, 1],
]

/**
 * 石を置いて裏返した新しい盤面を返す
 * @param {number[][]} board
 * @param {number} row
 * @param {number} col
 * @param {number} color - 1=black, 2=white
 * @returns {number[][]}
 */
export function placeStone(board, row, col, color) {
  const opponent = color === 1 ? 2 : 1
  const next = board.map(r => [...r])
  next[row][col] = color

  for (const [dr, dc] of DIRECTIONS) {
    const toFlip = []
    let r = row + dr
    let c = col + dc

    while (r >= 0 && r < 8 && c >= 0 && c < 8 && next[r][c] === opponent) {
      toFlip.push([r, c])
      r += dr
      c += dc
    }

    if (toFlip.length > 0 && r >= 0 && r < 8 && c >= 0 && c < 8 && next[r][c] === color) {
      for (const [fr, fc] of toFlip) {
        next[fr][fc] = color
      }
    }
  }

  return next
}

/**
 * 石の数をカウント
 * @param {number[][]} board
 * @returns {{ black: number, white: number }}
 */
export function countStones(board) {
  let black = 0
  let white = 0
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (board[row][col] === 1) black++
      else if (board[row][col] === 2) white++
    }
  }
  return { black, white }
}
