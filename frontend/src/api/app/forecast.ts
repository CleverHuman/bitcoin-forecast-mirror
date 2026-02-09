import axiosInstance from '@/stores/axios-api';

export type ForecastRequest = {
  from_date?: string;
  days?: number;
  no_signals?: boolean;
};

export type ForecastResponse = {
  timestamp: string;
  forecast_days: number;
  from_date?: string;
  data_points: number;
  forecast_file: string;
  signals_file: string;
  current_signal?: any;
  backtest?: any;
  error?: string;
};

export type ForecastHistoryItem = {
  filename: string;
  timestamp: string;
  from_date?: string;
  forecast_points: number;
  file_size: number;
};

export type ForecastHistoryResponse = {
  total: number;
  forecasts: ForecastHistoryItem[];
  error?: string;
};

const forecastEndpoint = '/forecast/run';
const historyEndpoint = '/forecast/history';
const fileEndpoint = '/forecast/file';

export type ForecastProgressEvent = {
  step: number;
  total: number;
  message: string;
};

export type ForecastSSECallbacks = {
  onProgress?: (progress: ForecastProgressEvent) => void;
  onComplete?: (result: ForecastResponse) => void;
  onError?: (error: string) => void;
};

export const runForecast = (
  request: ForecastRequest = {},
  callbacks: ForecastSSECallbacks
): (() => void) => {
  const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
  const url = `${baseURL}${forecastEndpoint}`;

  // Create abort controller for cleanup
  const abortController = new AbortController();

  const connectSSE = async () => {
    try {
      // Use fetch to POST, then create EventSource from response
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(request),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      let buffer = '';
      let isReading = true;

      while (isReading && !abortController.signal.aborted) {
        const { done, value } = await reader.read();

        if (done) {
          isReading = false;
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (separated by double newline)
        const messages = buffer.split('\n\n');

        // Keep the last incomplete message in buffer
        buffer = messages.pop() || '';

        for (const message of messages) {
          if (!message.trim()) continue;

          console.log('Received SSE message:', message);

          // Parse event type and data
          const lines = message.split('\n');
          let eventType = '';
          let data = '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.substring(7).trim();
            } else if (line.startsWith('data: ')) {
              data = line.substring(6).trim();
            }
          }

          if (eventType && data) {
            try {
              const parsedData = JSON.parse(data);
              console.log(`SSE Event: ${eventType}`, parsedData);

              if (eventType === 'progress' && callbacks.onProgress) {
                callbacks.onProgress(parsedData);
              } else if (eventType === 'complete' && callbacks.onComplete) {
                callbacks.onComplete(parsedData);
                isReading = false;
              } else if (eventType === 'error' && callbacks.onError) {
                callbacks.onError(parsedData.message || 'Unknown error');
                isReading = false;
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e, 'Raw data:', data);
            }
          }
        }
      }
    } catch (error: any) {
      if (callbacks.onError) {
        callbacks.onError(error.message || 'Connection failed');
      }
    }
  };

  connectSSE();

  // Return cleanup function
  return () => {
    abortController.abort();
  };
};

export const getForecastHistory = async (): Promise<ForecastHistoryResponse> => {
  const response = await axiosInstance.get<ForecastHistoryResponse>(
    historyEndpoint
  );
  return response.data;
};

export const getForecastFile = async (filename: string): Promise<string> => {
  const response = await axiosInstance.get<string>(
    `${fileEndpoint}/${filename}`,
    {
      responseType: 'text'
    }
  );
  return response.data;
};
