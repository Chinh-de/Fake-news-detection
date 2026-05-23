import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8000" });

api.interceptors.request.use((cfg) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("vifn_token");
    if (token) cfg.headers.Authorization = `Bearer ${token}`;
  }
  return cfg;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && typeof window !== "undefined") {
      // Only redirect if we're not already on the login page to avoid loops
      if (!window.location.pathname.startsWith("/login")) {
        localStorage.removeItem("vifn_token");
        localStorage.removeItem("vifn_user");
        window.location.replace("/login");
      }
    }
    return Promise.reject(err);
  }
);

export const authApi = {
  register: async (username: string, email: string, password: string) => {
    const { data } = await api.post("/api/auth/register", { username, email, password });
    return data;
  },
  login: async (username: string, password: string) => {
    const form = new FormData();
    form.append("username", username);
    form.append("password", password);
    const { data } = await api.post("/api/auth/token", form);
    return data;
  },
  me: async () => (await api.get("/api/auth/me")).data,
};

export const newsApi = {
  predict: async (payload: {
    text: string;
    max_length?: number;
    top_k_bm25?: number;
    top_k_web?: number;
    enable_web?: boolean;
    crawl_web?: boolean;
  }) => (await api.post("/api/news/predict", payload)).data,

  submit: async (payload: {
    title: string;
    content: string;
    user_label: string;
    source_url?: string;
  }) => (await api.post("/api/news/submit", payload)).data,

  mySubmissions: async () => (await api.get("/api/news/my")).data,
};

export const adminApi = {
  stats: async () => (await api.get("/api/admin/stats")).data,
  users: async () => (await api.get("/api/admin/users")).data,
  updateUser: async (id: number, data: { role?: string; is_active?: boolean }) =>
    (await api.put(`/api/admin/users/${id}`, data)).data,
  submissions: async (status?: string) => {
    const params = status ? { status } : {};
    return (await api.get("/api/admin/submissions", { params })).data;
  },
  getSubmission: async (id: number) => (await api.get(`/api/admin/submissions/${id}`)).data,
  approve: async (id: number, admin_note?: string) =>
    (await api.put(`/api/admin/submissions/${id}/approve`, { admin_note })).data,
  reject: async (id: number, admin_note?: string) =>
    (await api.put(`/api/admin/submissions/${id}/reject`, { admin_note })).data,
  startRetrain: async (submission_ids: number[]) =>
    (await api.post("/api/admin/retrain/start", { submission_ids })).data,
  retrainJobs: async () => (await api.get("/api/admin/retrain/jobs")).data,
  getRetrainJob: async (id: number) => (await api.get(`/api/admin/retrain/jobs/${id}`)).data,
};

export default api;
