export type BaseResponse<T> = {
  results: T;
  message?: string;
  count?: number;
  previous?: string;
  next?: string;
};
