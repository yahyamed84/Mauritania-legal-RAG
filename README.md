# RAG-WebSystem – Mauritania Legal Chatbot

**A Django-based web application integrating a Retrieval-Augmented Generation (RAG) pipeline with a locally hosted GGUF model and the Gemini API.**

---

## 📖 Overview

**RAG-WebSystem** is a scalable web framework that seamlessly integrates:

1. **A local GGUF model** (for on-premise inference)  
2. **Gemini API** (for external AI services)  
3. **A Django backend** (handling business logic, REST endpoints, and database interactions)  
4. **PostgreSQL** (for reliable data storage)  
5. **A responsive front-end** built with HTML, CSS, and JavaScript

Users submit queries through a clean web interface. The RAG core retrieves relevant context (local or remote), augments prompts, and generates responses. All interactions and metadata are recorded in PostgreSQL, making this architecture suitable for applications that require both local AI inference and external API calls.

---

## 🏛️ Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                                 User Interface                                │
│                        (HTML, CSS, JavaScript + AJAX)                         │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                 RAG System                                    │
│                       (Coordinating retrieval, augmentation,                  │
│                        and generation via local model & Gemini API)           │
└───────────────────────────────────────────────────────────────────────────────┘
            │                       │                         │
            ▼                       ▼                         ▼
┌──────────────────────┐   ┌──────────────────┐   ┌────────────────────────┐
│  Local GGUF Model    │   │  Gemini API      │   │    Django Backend      │
│ (on-prem inference)  │   │ (external AI)    │   │ (Views, REST endpoints,│
│                      │   │                  │   │  business logic)       │
└──────────────────────┘   └──────────────────┘   └────────────────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────────┐
                                              │    PostgreSQL DB     │
                                              │ (user queries, logs, │
                                              │  metadata, settings) │
                                              └──────────────────────┘
