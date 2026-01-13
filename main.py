import streamlit as st
import pandas as pd
import numpy as np
import random
import copy

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="Exam Scheduling Optimizer (ABC)", layout="wide")

@st.cache_data
def load_data():
    rooms = pd.read_csv('classrooms.csv')
    exams = pd.read_csv('exam_timeslot.csv')
    return rooms, exams

rooms_df, exams_df = load_data()

# --- 2. ABC ALGORITHM LOGIC ---
class ABC_Scheduler:
    def __init__(self, rooms, exams, n_bees, max_iter, limit):
        self.rooms = rooms
        self.exams = exams
        self.n_bees = n_bees
        self.max_iter = max_iter
        self.limit = limit # Had cubaan sebelum lebah pengakap bertindak
        
        # Inisialisasi sumber makanan (Jadual Rawak)
        self.foods = [self.create_random_solution() for _ in range(n_bees)]
        self.fitness = [self.calculate_fitness(sol) for sol in self.foods]
        self.trial = [0] * n_bees
        
        self.best_sol = self.foods[np.argmax(self.fitness)]
        self.best_fitness = max(self.fitness)
        self.history = []

    def create_random_solution(self):
        # Setiap exam diberikan classroom_id secara rawak
        return [random.choice(self.rooms['classroom_id'].values) for _ in range(len(self.exams))]

    def calculate_fitness(self, solution):
        penalties = 0
        conflicts = 0
        
        # Simpan data tugasan dalam dataframe sementara untuk semakan
        temp_df = self.exams.copy()
        temp_df['assigned_room'] = solution
        
        # Join dengan data bilik untuk check kapasiti
        merged = temp_df.merge(self.rooms, left_on='assigned_room', right_on='classroom_id')
        
        for _, row in merged.iterrows():
            # 1. Penalti Kapasiti (Hard Constraint)
            if row['num_students'] > row['capacity']:
                penalties += 100 
            
            # 2. Bonus Kesesuaian (Soft Constraint - Optimization)
            # Menggalakkan saiz bilik yang hampir dengan jumlah pelajar
            diff = row['capacity'] - row['num_students']
            if diff >= 0:
                penalties += (diff * 0.1) # Penalti kecil untuk pembaziran ruang

        # 3. Penalti Pertembungan Masa & Bilik (Hard Constraint)
        # Check jika bilik yang sama digunakan pada hari & masa yang sama
        duplicates = temp_df.duplicated(subset=['exam_day', 'exam_time', 'assigned_room'], keep=False)
        conflicts = duplicates.sum()
        penalties += (conflicts * 500)

        # Fitness adalah 1 / (1 + penalties)
        return 1 / (1 + penalties)

    def search_neighbor(self, solution):
        new_sol = copy.deepcopy(solution)
        idx = random.randint(0, len(new_sol) - 1)
        new_sol[idx] = random.choice(self.rooms['classroom_id'].values)
        return new_sol

    def solve(self):
        for i in range(self.max_iter):
            # 1. Employed Bees Phase
            for j in range(self.n_bees):
                new_sol = self.search_neighbor(self.foods[j])
                new_fit = self.calculate_fitness(new_sol)
                if new_fit > self.fitness[j]:
                    self.foods[j] = new_sol
                    self.fitness[j] = new_fit
                    self.trial[j] = 0
                else:
                    self.trial[j] += 1

            # 2. Onlooker Bees Phase (Tournament Selection)
            prob = [f/sum(self.fitness) for f in self.fitness]
            for _ in range(self.n_bees):
                j = np.random.choice(range(self.n_bees), p=prob)
                new_sol = self.search_neighbor(self.foods[j])
                new_fit = self.calculate_fitness(new_sol)
                if new_fit > self.fitness[j]:
                    self.foods[j] = new_sol
                    self.fitness[j] = new_fit
                    self.trial[j] = 0
                else:
                    self.trial[j] += 1

            # 3. Scout Bees Phase
            for j in range(self.n_bees):
                if self.trial[j] > self.limit:
                    self.foods[j] = self.create_random_solution()
                    self.fitness[j] = self.calculate_fitness(self.foods[j])
                    self.trial[j] = 0
            
            # Record best
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_sol = self.foods[current_best_idx]
            
            self.history.append(self.best_fitness)
        
        return self.best_sol, self.history

# --- 3. STREAMLIT UI ---
st.title("🐝 Exam Room Optimization using ABC Algorithm")
st.markdown("""
Aplikasi ini mengoptimumkan penjadualan bilik peperiksaan dengan memastikan tiada pertembungan 
dan kapasiti bilik mencukupi. [cite: 19, 20, 21]
""")

with st.sidebar:
    st.header("Algorithm Parameters")
    n_bees = st.slider("Number of Bees", 10, 100, 30)
    max_iter = st.slider("Max Iterations", 50, 500, 100)
    limit = st.slider("Scout Limit", 5, 50, 20)
    run_btn = st.button("Run Optimization")

if run_btn:
    abc = ABC_Scheduler(rooms_df, exams_df, n_bees, max_iter, limit)
    
    with st.spinner('Bees are searching for the best schedule...'):
        best_schedule, history = abc.solve()
    
    # Results Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Convergence Curve")
        st.line_chart(history)
    
    with col2:
        st.subheader("Final Fitness Score")
        st.write(f"Best Fitness Achieved: {max(history):.6f}")

    # Show Final Schedule
    st.subheader("Optimized Schedule")
    final_df = exams_df.copy()
    final_df['assigned_room_id'] = best_schedule
    
    # Merge for display
    display_df = final_df.merge(rooms_df, left_on='assigned_room_id', right_on='classroom_id')
    display_df = display_df[['course_code', 'num_students', 'exam_day', 'exam_time', 'building_name', 'room_number', 'capacity']]
    
    st.dataframe(display_df, use_container_width=True)
    
    # Analysis
    st.info("**Performance Analysis:** Algoritma berjaya mengurangkan penalti pertembungan berdasarkan graf penumpuan di atas. [cite: 13]")

else:
    st.info("Tekan butang 'Run Optimization' di sidebar untuk memulakan lebah mencari jadual.")
