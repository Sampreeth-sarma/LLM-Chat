import streamlit as st
import altair as alt
import pandas as pd
import time

class PerformanceVisualizer:
    def __init__(self):
        self.performance_data = []
        self.start_time = None
        self.token_count = 0
        
    def start_measurement(self):
        """Start measuring generation performance"""
        self.start_time = time.time()
        self.token_count = 0
        
    def add_token(self):
        """Count a new token generated"""
        self.token_count += 1
        
    def end_measurement(self, kv_cache_enabled):
        """End measurement and record performance data"""
        if self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        if self.token_count > 0:
            tokens_per_second = self.token_count / elapsed_time
            
            self.performance_data.append({
                "timestamp": time.time(),
                "tokens_per_second": tokens_per_second,
                "kv_cache": "Enabled" if kv_cache_enabled else "Disabled",
                "tokens": self.token_count,
                "time": elapsed_time
            })
        
        self.start_time = None
        self.token_count = 0
    
    def visualize_performance(self, container):
        """Create a performance visualization in the provided container"""
        if not self.performance_data:
            container.info("No performance data available yet. Generate some text first.")
            return
        
        df = pd.DataFrame(self.performance_data)
        
        # Create chart for tokens per second
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('tokens_per_second:Q', title='Tokens per Second'),
            color=alt.Color('kv_cache:N', title='KV Cache')
        ).properties(
            title='Generation Performance Over Time',
            width=600,
            height=300
        )
        
        container.altair_chart(chart, use_container_width=True)
        
        # Show average performance statistics
        avg_with_kv = df[df['kv_cache'] == 'Enabled']['tokens_per_second'].mean() if any(df['kv_cache'] == 'Enabled') else 0
        avg_without_kv = df[df['kv_cache'] == 'Disabled']['tokens_per_second'].mean() if any(df['kv_cache'] == 'Disabled') else 0
        
        col1, col2 = container.columns(2)
        
        with col1:
            st.metric("Avg. Speed with KV Cache", f"{avg_with_kv:.2f} tokens/sec")
            
        with col2:
            st.metric("Avg. Speed without KV Cache", f"{avg_without_kv:.2f} tokens/sec")
            
        if avg_with_kv > 0 and avg_without_kv > 0:
            speedup = (avg_with_kv / avg_without_kv - 1) * 100
            container.info(f"KV Cache provides a {speedup:.1f}% speedup on average")

    def clear_data(self):
        """Clear all performance data"""
        self.performance_data = []