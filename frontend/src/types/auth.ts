export interface CustomUser {
    id: number;
    first_name: string;
    last_name: string;
    email: string;
    avatar_url: string;
    get_display_name: string;
  }
  
  export interface LoginRequest {
    email?: string;
    password: string;
  }
  
  export interface LoginResponse {
    status: string;
    detail: string;
    jwt?: JWT;
    temp_otp_token?: string;
  }
  
  export interface JWT {
    access: string;
    refresh: string;
    user: CustomUser;
  }
  
  export interface RegisterRequest {
    username: string;
    email: string;
    password1: string;
    password2: string;
  }
  
  export interface OtpRequest {
    temp_otp_token: string;
    otp: string;
  }
  
  export interface PasswordChangeRequest {
    new_password1: string;
    new_password2: string;
  }
  
  export interface TokenRefreshRequest {
    refresh: string;
  }
  
  export interface TokenRefreshResponse {
    access: string;
  }
  
  export interface TokenVerifyRequest {
    token: string;
  }