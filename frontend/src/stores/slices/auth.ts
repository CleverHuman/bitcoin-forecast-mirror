import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { 
  LoginRequest, 
  LoginResponse, 
  RegisterRequest, 
  JWT, 
  OtpRequest, 
  PasswordChangeRequest,
  TokenRefreshRequest,
  TokenRefreshResponse,
  TokenVerifyRequest,
  CustomUser
} from '../../types/auth';
import axiosInstance from '../axios-api';
import { handleApiRequest } from '../utils';

interface AuthState {
  user: CustomUser | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  errors: Record<string, string[]>;
}

const initialState: AuthState = {
  user: null,
  accessToken: localStorage.getItem('accessToken'),
  refreshToken: localStorage.getItem('refreshToken'),
  isAuthenticated: !!localStorage.getItem('accessToken'),
  isLoading: false,
  errors: {},
};

export const login = createAsyncThunk<LoginResponse, LoginRequest, { rejectValue: Record<string, string[]> }>(
  'auth/login',
  async (credentials, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      if (!credentials.email || !credentials.password) {
        throw new Error('Email and password are required');
      }
      const auth = "Basic " + btoa(`${credentials.email}:${credentials.password}`);
    
      const response = await axiosInstance.post<LoginResponse>('/api/auth/login/', credentials, {
        headers: {
          "Authorization": auth
        }
      });
     
      return response.data;
    }, rejectWithValue);
  }
);

export const register = createAsyncThunk<JWT, RegisterRequest, { rejectValue: Record<string, string[]> }>(
  'auth/register',
  async (userData, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      const response = await axiosInstance.post<JWT>('/api/auth/register/', userData);
      return response.data;
    }, rejectWithValue);
  }
);

export const verifyOtp = createAsyncThunk<JWT, OtpRequest, { rejectValue: Record<string, string[]> }>(
  'auth/verifyOtp',
  async (otpData, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      const response = await axiosInstance.post<JWT>('/api/auth/verify-otp/', otpData);
      return response.data;
    }, rejectWithValue);
  }
);

export const logout = createAsyncThunk(
  'auth/logout',
  async (_, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      await axiosInstance.post('/api/auth/logout/');
    }, rejectWithValue);
  }
);

export const changePassword = createAsyncThunk<void, PasswordChangeRequest>(
  'auth/changePassword',
  async (passwordData, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      await axiosInstance.post('/api/auth/password/change/', passwordData);
    }, rejectWithValue);
  }
);

export const refreshToken = createAsyncThunk<TokenRefreshResponse, TokenRefreshRequest, { rejectValue: Record<string, string[]> }>(
  'auth/refreshToken',
  async (refreshData, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      const response = await axiosInstance.post<TokenRefreshResponse>('/api/auth/token/refresh/', refreshData);
      return response.data;
    }, rejectWithValue);
  }
);

export const verifyToken = createAsyncThunk<void, TokenVerifyRequest, { rejectValue: Record<string, string[]> }>(
  'auth/verifyToken',
  async (tokenData, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      await axiosInstance.post('/api/auth/token/verify/', tokenData);
    }, rejectWithValue);
  }
);

export const fetchUser = createAsyncThunk<CustomUser, void, { rejectValue: Record<string, string[]> }>(
  'auth/fetchUser',
  async (_, { rejectWithValue }) => {
    return handleApiRequest(async () => {
      const response = await axiosInstance.get<CustomUser>('/api/auth/user/');
      return response.data;
    }, rejectWithValue);
  }
);

export const refreshAccessToken = createAsyncThunk<TokenRefreshResponse, void, { rejectValue: Record<string, string[]> }>(
  'auth/refreshAccessToken',
  async (_, { getState, rejectWithValue }) => {
    const { auth } = getState() as { auth: AuthState };
    const refreshToken = auth.refreshToken;

    if (!refreshToken) {
      return rejectWithValue({ non_field_errors: ['No refresh token available'] });
    }

    return handleApiRequest(async () => {
      const response = await axiosInstance.post<TokenRefreshResponse>('/api/auth/token/refresh/', { refresh: refreshToken });
      return response.data;
    }, rejectWithValue);
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearErrors: (state) => {
      state.errors = {};
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(login.pending, (state) => {
        state.isLoading = true;
        state.errors = {};
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false;
        if (action.payload.jwt) {
          state.user = action.payload.jwt.user;
          state.accessToken = action.payload.jwt.access;
          state.refreshToken = action.payload.jwt.refresh;
          state.isAuthenticated = true;
          localStorage.setItem('accessToken', action.payload.jwt.access);
          localStorage.setItem('refreshToken', action.payload.jwt.refresh);
        }
        state.errors = {};
      })
      .addCase(login.rejected, (state, action: PayloadAction<Record<string, string[]> | undefined>) => {
        state.isLoading = false;
        if (action.payload) {
          state.errors = action.payload;
        } else {
          state.errors = { non_field_errors: ['Login failed'] };
        }
      })
      .addCase(register.pending, (state) => {
        state.isLoading = true;
        state.errors = {};
      })
      .addCase(register.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.accessToken = action.payload.access;
        state.refreshToken = action.payload.refresh;
        state.isAuthenticated = true;
        localStorage.setItem('accessToken', action.payload.access);
        localStorage.setItem('refreshToken', action.payload.refresh);
        state.errors = {};
      })
      .addCase(register.rejected, (state, action: PayloadAction<Record<string, string[]> | undefined>) => {
        state.isLoading = false;
        if (action.payload) {
          state.errors = action.payload;
        } else {
          state.errors = { non_field_errors: ['Registration failed'] };
        }
      })
      .addCase(verifyOtp.pending, (state) => {
        state.isLoading = true;
        state.errors = {};
      })
      .addCase(verifyOtp.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload.user;
        state.accessToken = action.payload.access;
        state.refreshToken = action.payload.refresh;
        state.isAuthenticated = true;
        localStorage.setItem('accessToken', action.payload.access);
        localStorage.setItem('refreshToken', action.payload.refresh);
        state.errors = {};
      })
      .addCase(verifyOtp.rejected, (state, action: PayloadAction<Record<string, string[]> | undefined>) => {
        state.isLoading = false;
        if (action.payload) {
          state.errors = action.payload;
        } else {
          state.errors = { non_field_errors: ['OTP verification failed'] };
        }
      })
      .addCase(logout.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logout.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.accessToken = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
      })
      .addCase(logout.rejected, (state) => {
        state.isLoading = false;
        state.errors = { non_field_errors: ['Logout failed'] };
      })
      .addCase(refreshToken.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(refreshToken.fulfilled, (state, action) => {
        state.isLoading = false;
        state.accessToken = action.payload.access;
      })
      .addCase(refreshToken.rejected, (state) => {
        state.isLoading = false;
        state.errors = { non_field_errors: ['Token refresh failed'] };
      })
      .addCase(fetchUser.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
      })
      .addCase(fetchUser.rejected, (state) => {
        state.isLoading = false;
        state.errors = { non_field_errors: ['Failed to fetch user data'] };
      })
      .addCase(refreshAccessToken.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(refreshAccessToken.fulfilled, (state, action) => {
        state.isLoading = false;
        state.accessToken = action.payload.access;
        state.isAuthenticated = true;
        localStorage.setItem('accessToken', action.payload.access);
      })
      .addCase(refreshAccessToken.rejected, (state) => {
        state.isLoading = false;
        state.user = null;
        state.accessToken = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        state.errors = { non_field_errors: ['Session expired, please login again'] };
      });
  },
});

export const { clearErrors } = authSlice.actions;
export default authSlice.reducer;