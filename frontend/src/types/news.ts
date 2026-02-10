import { BaseResponse } from "./base-response";

// "id": 0,
// "website": 0,
// "website_name": "string",
// "title": "string",
// "link": "string",
// "description": "string",
// "pub_date": "2024-11-25T09:36:58.892Z",
// "category": "string",
// "created_at": "2024-11-25T09:36:58.892Z"

export type Media = {
  id: number;
  media?: string;
  media_type?: number;
  medium?: string;
}

export type NewsItem = {
  id: number;
  title: string;
  link: string;
  pub_date: string;
  description: string;
  key_phrases: NewsKeyPhrase[];
  entities: NewsEntity[];
  media: Media[];
  categories: NewsCategory[];
  article_text: string;
  sentiment: string;
  sentiment_value: number;
  sentiment_confidence: number;
  website: number;
  website_name: string;
  created_at: string;
  ai_summary: string;
  hash_value: string;
  creator: string;
}

export type Website = {
  id?: number | 0;
  website_name: string;
  url: string;
  rss_feed_url: string;
  created_at: string;
}


export type NewsCategory = {
  name: string;
  id: number;
}

export type KeyPhrase = {
  id: number;
  phrase: string;
}

export type NewsKeyPhrase = {
  id: number;
  news: number;
  key_phrase: KeyPhrase;
}

export type EntityType = {
  id: number;
  name: string;
}

export type Entity = {
  id: number;
  entity: string;
  entity_type: EntityType;
}

export type NewsEntity = {
  id: number;
  news: number;
  entity: Entity;
}




export type NewsResponse = BaseResponse<NewsItem[]>;
export type NewsCategoriesResponse = BaseResponse<NewsCategory[]>;
export type NewsWebsitesResponse = BaseResponse<Website[]>;



