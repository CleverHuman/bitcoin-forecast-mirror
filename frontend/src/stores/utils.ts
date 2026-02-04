import { AxiosError } from "axios";

export const handleApiRequest = async <T>(
    request: () => Promise<T>,
    rejectWithValue: (value: Record<string, string[]>) => any
  ): Promise<T> => {
    try {
      return await request();
    } catch (err) {
      if (err instanceof AxiosError && err.response?.data) {
        return rejectWithValue(err.response.data as Record<string, string[]>);
      } else if (err instanceof Error) {
        return rejectWithValue({ non_field_errors: [err.message] });
      }
      return rejectWithValue({ non_field_errors: ['An unexpected error occurred'] });
    }
  };
  