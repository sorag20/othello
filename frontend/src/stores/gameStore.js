import { defineStore } from 'pinia'
import { gameApi } from '../api/gameApi.js'
import { placeStone } from '../utils/boardUtils.js'

export const useGameStore = defineStore('game', {
  state: () => ({
    gameId: null,
    board: [],
    currentTurn: 'black',
    legalMoves: [],
    stoneCount: { black: 2, white: 2 },
    status: null,
    result: null,
    lastMove: null,
    flippedStones: [],
    isLoading: false,
    error: null,
    isThinking: false,
    passed: false,
    // Opponent settings
    opponentType: 'internal',
    edaxLevel: 1,
    edaxAvailable: false,
    showSetup: true,
  }),

  getters: {
    isGameOver: (state) => state.status === 'finished',
    isPlayerTurn: (state) => state.currentTurn === 'black',
    isAITurn: (state) => state.currentTurn === 'white'
  },

  actions: {
    /**
     * Edax利用可否をサーバーから取得してキャッシュ
     */
    async fetchEdaxAvailable() {
      this.edaxAvailable = await gameApi.checkEdaxAvailable()
    },

    /**
     * 設定ダイアログを表示して次のゲームを準備する
     */
    showSetupDialog() {
      this.showSetup = true
    },

    /**
     * 新規ゲームを開始
     * @param {object} [options]
     * @param {string} [options.opponentType]
     * @param {number} [options.edaxLevel]
     */
    async startNewGame({ opponentType, edaxLevel } = {}) {
      if (opponentType !== undefined) this.opponentType = opponentType
      if (edaxLevel !== undefined) this.edaxLevel = edaxLevel

      this.showSetup = false
      this.isLoading = true
      this.error = null
      this.passed = false

      try {
        const data = await gameApi.createGame({
          opponentType: this.opponentType,
          edaxLevel: this.edaxLevel,
        })
        this.gameId = data.id
        this.board = data.board
        this.currentTurn = data.current_turn
        this.legalMoves = data.legal_moves
        this.stoneCount = data.stone_count
        this.status = data.status
        this.result = data.result || null
        this.lastMove = null
        this.flippedStones = []
        this.isThinking = false
      } catch (error) {
        this.error = error.message || 'ゲームの作成に失敗しました'
        this.showSetup = true
      } finally {
        this.isLoading = false
      }
    },

    /**
     * 手を打つ（プレイヤーとAIの応答両方を処理）
     * @param {number} row
     * @param {number} col
     */
    async makeMove(row, col) {
      if (!this.gameId) return

      this.isLoading = true
      this.error = null
      this.passed = false

      try {
        const data = await gameApi.makeMove(this.gameId, row, col)
        this.board = data.board
        this.currentTurn = data.current_turn
        this.legalMoves = data.legal_moves
        this.stoneCount = data.stone_count
        this.status = data.status
        this.result = data.result || null
        this.lastMove = data.move || null
        this.flippedStones = data.flipped || []

        // パス判定: AIターンで合法手がない場合
        if (data.passed) {
          this.passed = true
        }
      } catch (error) {
        this.error = error.message || '手の送信に失敗しました'
      } finally {
        this.isLoading = false
      }
    },

    /**
     * ゲーム状態を取得
     * @param {string} gameId
     */
    async fetchGameState(gameId) {
      this.isLoading = true
      this.error = null

      try {
        const data = await gameApi.getGame(gameId)
        this.gameId = data.id
        this.board = data.board
        this.currentTurn = data.current_turn
        this.legalMoves = data.legal_moves
        this.stoneCount = data.stone_count
        this.status = data.status
        this.result = data.result || null
      } catch (error) {
        this.error = error.message || 'ゲーム状態の取得に失敗しました'
      } finally {
        this.isLoading = false
      }
    },

    /**
     * プレイヤーの手を楽観的に盤面へ反映（API応答を待たずに石を置く）
     * @param {number} row
     * @param {number} col
     */
    applyOptimisticMove(row, col) {
      this.board = placeStone(this.board, row, col, 1)  // プレイヤーは常に黒 (1)
      this.legalMoves = []
    },

    /**
     * AI思考中状態を設定
     * @param {boolean} value
     */
    setThinking(value) {
      this.isThinking = value
    },

    /**
     * エラーをクリア
     */
    clearError() {
      this.error = null
    },

    /**
     * ストアをリセット
     */
    reset() {
      this.gameId = null
      this.board = []
      this.currentTurn = 'black'
      this.legalMoves = []
      this.stoneCount = { black: 2, white: 2 }
      this.status = null
      this.result = null
      this.lastMove = null
      this.flippedStones = []
      this.isLoading = false
      this.error = null
      this.isThinking = false
      this.passed = false
      this.showSetup = true
    }
  }
})
