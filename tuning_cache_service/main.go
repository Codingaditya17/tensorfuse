// tuning_cache_service — a small HTTP service that stores autotuning
// results (which backend/unroll won, for a given op-signature + shape)
// so multiple compiler processes/machines can share ONE tuning
// database instead of each re-running the search cold.
//
// This is the same idea as autotune.py's local JSON file, made
// networked: GET to look up a cached winner, POST to record one.
// Persists to disk so a restart doesn't lose the tuning log.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
)

type TuningEntry struct {
	Backend      string      `json:"backend"`
	Unroll       int         `json:"unroll"`
	TimeMs       float64     `json:"time_ms"`
	AllCandidates interface{} `json:"all_candidates,omitempty"`
}

type Store struct {
	mu      sync.RWMutex
	entries map[string]TuningEntry
	path    string
}

func NewStore(path string) *Store {
	s := &Store{entries: make(map[string]TuningEntry), path: path}
	s.load()
	return s
}

func (s *Store) load() {
	data, err := os.ReadFile(s.path)
	if err != nil {
		log.Printf("no existing store at %s (starting fresh): %v", s.path, err)
		return
	}
	var entries map[string]TuningEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		log.Printf("failed to parse existing store: %v", err)
		return
	}
	s.mu.Lock()
	s.entries = entries
	s.mu.Unlock()
	log.Printf("loaded %d cached tuning entries from %s", len(entries), s.path)
}

func (s *Store) persist() {
	s.mu.RLock()
	data, err := json.MarshalIndent(s.entries, "", "  ")
	s.mu.RUnlock()
	if err != nil {
		log.Printf("marshal error: %v", err)
		return
	}
	if err := os.WriteFile(s.path, data, 0644); err != nil {
		log.Printf("write error: %v", err)
	}
}

func (s *Store) Get(key string) (TuningEntry, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.entries[key]
	return e, ok
}

func (s *Store) Put(key string, e TuningEntry) {
	s.mu.Lock()
	s.entries[key] = e
	s.mu.Unlock()
	s.persist()
}

func (s *Store) All() map[string]TuningEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]TuningEntry, len(s.entries))
	for k, v := range s.entries {
		out[k] = v
	}
	return out
}

func key(sig, cols string) string {
	return fmt.Sprintf("%s|cols=%s", sig, cols)
}

func main() {
	port := os.Getenv("TUNING_CACHE_PORT")
	if port == "" {
		port = "8787"
	}
	dbPath := os.Getenv("TUNING_CACHE_DB")
	if dbPath == "" {
		dbPath = "tuning_cache_db.json"
	}
	store := NewStore(dbPath)

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	mux.HandleFunc("/tuning", func(w http.ResponseWriter, r *http.Request) {
		sig := r.URL.Query().Get("sig")
		cols := r.URL.Query().Get("cols")
		if sig == "" || cols == "" {
			http.Error(w, "sig and cols query params required", http.StatusBadRequest)
			return
		}
		k := key(sig, cols)

		switch r.Method {
		case http.MethodGet:
			entry, ok := store.Get(k)
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(entry)

		case http.MethodPost:
			var entry TuningEntry
			if err := json.NewDecoder(r.Body).Decode(&entry); err != nil {
				http.Error(w, "invalid JSON body: "+err.Error(), http.StatusBadRequest)
				return
			}
			store.Put(k, entry)
			log.Printf("stored tuning result: %s -> backend=%s unroll=%d time_ms=%.4f",
				k, entry.Backend, entry.Unroll, entry.TimeMs)
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, "stored")

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/all", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(store.All())
	})

	addr := ":" + port
	log.Printf("tuning_cache_service listening on %s (db: %s)", addr, dbPath)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
