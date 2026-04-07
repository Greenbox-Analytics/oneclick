export interface Artist {
  id: string;
  name: string;
}

export interface Project {
  id: string;
  name: string;
  artist_id: string;
}

export interface Contract {
  id: string;
  file_name: string;
  project_id: string;
  folder_category?: string;
}

export interface RoyaltySplitData {
  party: string;
  percentage: number;
}

export interface RoyaltySplitsByType {
  streaming?: RoyaltySplitData[];
  publishing?: RoyaltySplitData[];
  mechanical?: RoyaltySplitData[];
  sync?: RoyaltySplitData[];
  master?: RoyaltySplitData[];
  performance?: RoyaltySplitData[];
  general?: RoyaltySplitData[];
}

export interface ExtractedContractData {
  royalty_splits?: RoyaltySplitsByType;
  payment_terms?: string;
  parties?: string[];
  advances?: string;
  term_length?: string;
  [key: string]: unknown;
}

export interface ArtistDataExtracted {
  bio?: string;
  social_media?: Record<string, string>;
  streaming_links?: Record<string, string>;
  genres?: string[];
  email?: string;
}

export interface ArtistDiscussed {
  id: string;
  name: string;
  data_extracted: ArtistDataExtracted;
}

export interface ContractDiscussed {
  id: string;
  name: string;
  data_extracted: ExtractedContractData;
}

export interface ContextSwitch {
  timestamp: string;
  type: 'artist' | 'project' | 'contract';
  from: string;
  to: string;
}

export interface ConversationContext {
  session_id: string;
  artist: { id: string; name: string } | null;
  artists_discussed: ArtistDiscussed[];
  project: { id: string; name: string } | null;
  contracts_discussed: ContractDiscussed[];
  context_switches: ContextSwitch[];
}
