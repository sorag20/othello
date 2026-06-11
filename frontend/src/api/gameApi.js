import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json'
  }
})

/**
 * APIエラーを統一フォーマットにラップ
 */
function handleError(error) {
  if (error.response) {
    // サーバーからのエラーレスポンス
    const data = error.response.data
    const apiError = new Error(
      data?.error?.message || `Request failed with status ${error.response.status}`
    )
    apiError.code = data?.error?.code || 'UNKNOWN_ERROR'
    apiError.status = error.response.status
    throw apiError
  } else if (error.request) {
    // ネットワークエラー
    const networkError = new Error('ネットワーク接続に失敗しました')
    networkError.code = 'NETWORK_ERROR'
    networkError.status = null
    throw networkError
  } else {
    throw error
  }
}

/**
 * ゲームAPI クライアント
 */
export const gameApi = {
  /**
   * 新規ゲームを作成
   * @returns {Promise<object>} GameResponse
   */
  async createGame() {
    try {
      const response = await apiClient.post('/games')
      return response.data
    } catch (error) {
      handleError(error)
    }
  },

  /**
   * ゲーム状態を取得
   * @param {string} gameId
   * @returns {Promise<object>} GameResponse
   */
  async getGame(gameId) {
    try {
      const response = await apiClient.get(`/games/${gameId}`)
      return response.data
    } catch (error) {
      handleError(error)
    }
  },

  /**
   * 手を打つ
   * @param {string} gameId
   * @param {number} row
   * @param {number} col
   * @returns {Promise<object>} MoveResponse
   */
  async makeMove(gameId, row, col) {
    try {
      const response = await apiClient.post(`/games/${gameId}/moves`, { row, col })
      return response.data
    } catch (error) {
      handleError(error)
    }
  },

  /**
   * 対局履歴一覧を取得
   * @param {number} [page=1]
   * @returns {Promise<object>} HistoryResponse
   */
  async getHistory(page = 1) {
    try {
      const response = await apiClient.get('/games', { params: { page } })
      return response.data
    } catch (error) {
      handleError(error)
    }
  },

  /**
   * リプレイデータを取得
   * @param {string} gameId
   * @returns {Promise<object>} ReplayResponse
   */
  async getReplay(gameId) {
    try {
      const response = await apiClient.get(`/games/${gameId}/replay`)
      return response.data
    } catch (error) {
      handleError(error)
    }
  }
}
