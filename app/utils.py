import streamlit as st
import pandas as pd


def is_exam_period(date, exam_periods):
    for start, end in exam_periods:
        if start <= date <= end:
            return True
    return False

# Preprocess data function
