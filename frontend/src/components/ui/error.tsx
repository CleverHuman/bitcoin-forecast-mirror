// import UnauthorizedView from "@/pages/feed/components/unauthorized-view";
import { AxiosError } from "axios";

interface ErrorMessageProps {
  error: any | null;
  className?: string;
}

export function ErrorMessage({ error, className = '' }: ErrorMessageProps) {
  if (!error) return null;

  let errorMessage = error;
  let errorCode = null;

  if (error instanceof AxiosError) {
    errorMessage = error.response?.data?.message || error.message || 'An error occurred';
    errorCode = error.response?.status || null;
  }

  if (errorCode === 401 || errorCode === 403) {
    // return <UnauthorizedView />
  }


  return (
    <div className={`p-4 text-red-800 bg-red-100 rounded-md ${className}`}>
      <p>{errorMessage.toString()}</p>
    </div>
  );
}