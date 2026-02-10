import axiosInstance from "@/stores/axios-api";
import { NewsCategoriesResponse, NewsItem, NewsResponse, NewsWebsitesResponse } from "@/types/news";

const newsEndpoint = '/api/news/';

export type NewsParams = {
  category?: string | number | null;
  category_name?: string | null;
  created_at__gte?: number;
  created_at__lte?: number;
  page: number;
  pub_date__gte?: string;
  pub_date__lte?: string;
  title__icontains?: string;
  website?: string;
  websites?: string[] | string;
}

export type NewsCategoryParams = {
  name?: string | null;
  page: number;
}

export type NewsArticleTextParams = {
  refresh?: boolean;
  cache_days?: number;
  new_proxy?: boolean
}

export type NewsWebsiteParams = {
  search?: string | null;
  website_name__icontains?: string | null;
  page: number;
}

export const getNews = async (params: NewsParams) => {
  if(params.websites && Array.isArray(params.websites)) {
    params.websites = params.websites.join(',');
  }
  const response = await axiosInstance.get<NewsResponse>(newsEndpoint, { params });
  return response.data;
};

export const getNewsById = async (id: string) => {
  const response = await axiosInstance.get<NewsItem>(`${newsEndpoint}${id}`);
  return response.data;
};

export const getNewsCategories = async (params: NewsCategoryParams) => {
  const response = await axiosInstance.get<NewsCategoriesResponse>(`${newsEndpoint}categories/`, { params });
  return response.data;
};


export const getNewsWebsites = async (params: NewsWebsiteParams) => {
  if (params.search) {
    params.website_name__icontains = params.search;
    params.search = undefined;
  }
  const response = await axiosInstance.get<NewsWebsitesResponse>(`${newsEndpoint}websites/`, { params });
  return response.data;
};

export const getNewsArticleText = async (id: string | number, params?: NewsArticleTextParams) => {
  const response = await axiosInstance.get<NewsItem>(`${newsEndpoint}${id}/article_text`, { params });
  return response.data;
};

export const getNewsSentiment = async (id: string | number) => {
  const response = await axiosInstance.get<NewsItem>(`${newsEndpoint}${id}/sentiments`);
  return response.data;
};
