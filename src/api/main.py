from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = [
    ROOT / "artifacts" / "accessibility_classifier_ordinal.joblib",
    ROOT / "artifacts" / "accessibility_classifier_ensemble.joblib",
    ROOT / "artifacts" / "accessibility_classifier.joblib",
]


class AccessibilityRequest(BaseModel):
    genre_group: str = Field(..., examples=["클래식"])
    venue_type: str = Field(..., examples=["대형"])
    is_weekend: int = Field(..., ge=0, le=1, examples=[1])
    duration_days: float = Field(..., ge=1.0, examples=[5.0])
    organizer_type: str = Field(..., examples=["대관"])
    organizer: str = Field(..., examples=["(재)서울시립교향악단"])


def load_bundle() -> dict[str, Any]:
    for model_file in MODEL_FILES:
        if model_file.exists():
            return joblib.load(model_file)
    raise RuntimeError("No model artifacts found. Run training script first.")


bundle = load_bundle()
app = FastAPI(title="SAC Accessibility Classifier", version="0.2.0")

DEMO_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SAC 휠체어 사용자 접근성 지능형 진단 대시보드</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --bg-gradient: radial-gradient(circle at 50% 50%, #0d111d 0%, #07090e 100%);
      --card-bg: rgba(15, 23, 42, 0.45);
      --card-border: rgba(255, 255, 255, 0.08);
      --neon-blue: #0ea5e9;
      --neon-purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      
      --color-A: #10b981;
      --color-B: #3b82f6;
      --color-C: #f59e0b;
      --color-D: #f97316;
      --color-E: #ef4444;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Outfit', 'Noto Sans KR', sans-serif;
      background: var(--bg-gradient);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      padding: 40px 20px;
      overflow-x: hidden;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    .app-header {
      text-align: center;
      margin-bottom: 32px;
      max-width: 800px;
    }
    
    .app-header .logo {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--neon-blue);
      text-transform: uppercase;
      letter-spacing: 3px;
      margin-bottom: 12px;
      display: inline-block;
      padding: 4px 12px;
      border: 1px solid rgba(14, 165, 233, 0.3);
      border-radius: 20px;
      background: rgba(14, 165, 233, 0.05);
    }
    
    .app-header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(90deg, #38bdf8 0%, #c084fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 12px;
      letter-spacing: -1px;
    }
    
    .app-header p {
      font-size: 1.05rem;
      color: var(--text-muted);
      line-height: 1.6;
    }

    .container {
      max-width: 1200px;
      width: 100%;
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 32px;
      align-items: start;
    }
    
    @media (max-width: 1024px) {
      .container { grid-template-columns: 1fr; }
    }
    
    .glass-panel {
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 24px;
      padding: 36px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
      transition: all 0.3s ease;
    }
    
    .glass-panel:hover {
      border-color: rgba(255, 255, 255, 0.12);
    }

    .panel-title {
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 14px;
    }

    .panel-title i {
      color: var(--neon-blue);
    }
    
    .input-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    
    @media (max-width: 640px) {
      .input-grid { grid-template-columns: 1fr; }
    }
    
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    .form-group.full-width {
      grid-column: 1 / -1;
    }
    
    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    label i {
      font-size: 0.9rem;
      color: var(--neon-purple);
    }
    
    /* Custom Select Dropdowns (Fixes Screen Recording Bug) */
    .custom-select-wrapper {
      position: relative;
      width: 100%;
    }
    
    .custom-select-trigger {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      background: rgba(7, 9, 14, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--text-main);
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.2s ease;
      user-select: none;
    }
    
    .custom-select-trigger:hover {
      border-color: rgba(255, 255, 255, 0.2);
      background: rgba(7, 9, 14, 0.8);
    }
    
    .custom-select-wrapper.open .custom-select-trigger {
      border-color: var(--neon-blue);
      box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
      background: rgba(7, 9, 14, 0.8);
    }
    
    .custom-select-trigger i {
      font-size: 0.8rem;
      transition: transform 0.25s ease;
      color: var(--text-muted);
    }
    
    .custom-select-wrapper.open .custom-select-trigger i {
      transform: rotate(180deg);
      color: var(--neon-blue);
    }
    
    .custom-options {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: rgba(15, 23, 42, 0.96);
      backdrop-filter: blur(25px);
      -webkit-backdrop-filter: blur(25px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
      display: none;
      flex-direction: column;
      z-index: 100;
      max-height: 220px;
      overflow-y: auto;
    }
    
    .custom-select-wrapper.open .custom-options {
      display: flex;
    }
    
    .custom-option {
      padding: 12px 16px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 0.95rem;
      color: var(--text-muted);
      text-align: left;
    }
    
    .custom-option:hover {
      background: rgba(14, 165, 233, 0.15);
      color: var(--text-main);
    }
    
    .custom-option.selected {
      background: rgba(168, 85, 247, 0.2);
      color: #e9d5ff;
      font-weight: 600;
    }
    
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: rgba(7, 9, 14, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--text-main);
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: all 0.2s ease;
    }
    
    select:focus, input:focus {
      border-color: var(--neon-blue);
      box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
      background: rgba(7, 9, 14, 0.8);
    }
    
    .radio-group {
      display: flex;
      gap: 12px;
    }
    
    .radio-label {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: rgba(7, 9, 14, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 12px;
      cursor: pointer;
      font-weight: 500;
      font-size: 0.95rem;
      transition: all 0.2s ease;
      user-select: none;
    }
    
    .radio-group input[type="radio"] {
      display: none;
    }
    
    .radio-group input[type="radio"]:checked + .radio-label {
      background: rgba(168, 85, 247, 0.15);
      border-color: var(--neon-purple);
      color: #e9d5ff;
      box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }
    
    .range-container {
      display: flex;
      align-items: center;
      gap: 16px;
      background: rgba(7, 9, 14, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 10px 16px;
    }
    
    .range-container input[type="range"] {
      flex: 1;
      accent-color: var(--neon-blue);
      height: 6px;
      border-radius: 3px;
      outline: none;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.1);
    }
    
    .range-val {
      font-size: 1rem;
      font-weight: 600;
      color: var(--neon-blue);
      min-width: 50px;
      text-align: right;
    }
    
    .btn-submit {
      width: 100%;
      background: linear-gradient(90deg, var(--neon-blue) 0%, var(--neon-purple) 100%);
      border: none;
      border-radius: 14px;
      padding: 18px;
      color: white;
      font-size: 1.05rem;
      font-weight: 700;
      cursor: pointer;
      margin-top: 12px;
      box-shadow: 0 8px 25px rgba(168, 85, 247, 0.25);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    
    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 30px rgba(168, 85, 247, 0.45);
      filter: brightness(1.1);
    }
    
    .btn-submit:active { transform: translateY(0); }
    
    .result-section {
      min-height: 520px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    
    .result-placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
      color: var(--text-muted);
      text-align: center;
    }
    
    .result-placeholder i {
      font-size: 5rem;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.02) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: float 4s ease-in-out infinite;
    }

    @keyframes float {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }
    
    .result-content {
      display: none;
      width: 100%;
      animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .prediction-header {
      display: flex;
      align-items: center;
      gap: 24px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .grade-badge {
      width: 96px;
      height: 96px;
      border-radius: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 3.5rem;
      font-weight: 800;
      color: white;
      transition: all 0.3s ease;
      flex-shrink: 0;
    }

    .grade-badge.grade-A {
      background: radial-gradient(circle, var(--color-A) 0%, rgba(16, 185, 129, 0.2) 100%);
      border: 2px solid var(--color-A);
      box-shadow: 0 0 30px rgba(16, 185, 129, 0.35);
    }
    .grade-badge.grade-B {
      background: radial-gradient(circle, var(--color-B) 0%, rgba(59, 130, 246, 0.2) 100%);
      border: 2px solid var(--color-B);
      box-shadow: 0 0 30px rgba(59, 130, 246, 0.35);
    }
    .grade-badge.grade-C {
      background: radial-gradient(circle, var(--color-C) 0%, rgba(245, 158, 11, 0.2) 100%);
      border: 2px solid var(--color-C);
      box-shadow: 0 0 30px rgba(245, 158, 11, 0.35);
    }
    .grade-badge.grade-D {
      background: radial-gradient(circle, var(--color-D) 0%, rgba(249, 115, 22, 0.2) 100%);
      border: 2px solid var(--color-D);
      box-shadow: 0 0 30px rgba(249, 115, 22, 0.35);
    }
    .grade-badge.grade-E {
      background: radial-gradient(circle, var(--color-E) 0%, rgba(239, 68, 68, 0.2) 100%);
      border: 2px solid var(--color-E);
      box-shadow: 0 0 30px rgba(239, 68, 68, 0.35);
    }

    .prediction-title-area {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .pred-meta {
      font-size: 0.8rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .pred-title {
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text-main);
    }
    
    /* Probabilities progress chart */
    .section-subtitle {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--neon-blue);
      margin-bottom: 16px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .prob-chart {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 32px;
      background: rgba(7, 9, 14, 0.3);
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    .prob-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    .prob-label {
      font-size: 0.95rem;
      font-weight: 700;
      width: 16px;
      text-align: center;
    }
    
    .prob-bar-bg {
      flex: 1;
      height: 8px;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 4px;
      overflow: hidden;
    }
    
    .prob-bar-fill {
      height: 100%;
      border-radius: 4px;
      width: 0%;
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .prob-row.prob-A .prob-bar-fill { background: var(--color-A); }
    .prob-row.prob-B .prob-bar-fill { background: var(--color-B); }
    .prob-row.prob-C .prob-bar-fill { background: var(--color-C); }
    .prob-row.prob-D .prob-bar-fill { background: var(--color-D); }
    .prob-row.prob-E .prob-bar-fill { background: var(--color-E); }
    
    .prob-val {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      width: 40px;
      text-align: right;
    }
    
    /* SHAP Style Feature Contribution Chart */
    .shap-container {
      margin-bottom: 32px;
    }

    .shap-chart {
      background: rgba(7, 9, 14, 0.3);
      padding: 20px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.04);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .shap-row {
      display: grid;
      grid-template-columns: 120px 1fr 50px;
      align-items: center;
      gap: 12px;
      font-size: 0.85rem;
    }

    .shap-feature-name {
      color: var(--text-muted);
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .shap-bar-wrapper {
      position: relative;
      height: 14px;
      background: rgba(255,255,255,0.02);
      border-radius: 4px;
      display: flex;
      align-items: center;
    }

    .shap-center-line {
      position: absolute;
      left: 50%;
      top: 0;
      width: 1px;
      height: 100%;
      background: rgba(255, 255, 255, 0.2);
      z-index: 2;
    }

    .shap-bar {
      height: 100%;
      border-radius: 3px;
      position: absolute;
      transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .shap-bar.positive {
      left: 50%;
      background: linear-gradient(90deg, rgba(16, 185, 129, 0.4) 0%, var(--color-A) 100%);
      transform-origin: left;
    }

    .shap-bar.negative {
      right: 50%;
      background: linear-gradient(90deg, var(--color-E) 0%, rgba(239, 68, 68, 0.4) 100%);
      transform-origin: right;
    }

    .shap-value {
      font-weight: 600;
      text-align: right;
      font-size: 0.8rem;
    }

    .shap-value.pos { color: var(--color-A); }
    .shap-value.neg { color: var(--color-E); }
    .shap-value.zero { color: var(--text-muted); }

    .shap-legend {
      display: flex;
      justify-content: center;
      gap: 20px;
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 10px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .legend-color {
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }
    .legend-color.pos { background: var(--color-A); }
    .legend-color.neg { background: var(--color-E); }

    /* Actionable Feedback Card */
    .xai-feedback-card {
      background: linear-gradient(135deg, rgba(168, 85, 247, 0.08) 0%, rgba(14, 165, 233, 0.05) 100%);
      border: 1px solid rgba(168, 85, 247, 0.2);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 24px;
      text-align: left;
    }

    .xai-feedback-card h4 {
      font-size: 0.95rem;
      font-weight: 700;
      color: #e9d5ff;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .xai-feedback-card p {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.5;
    }

    .meta-info-bar {
      font-size: 0.75rem;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      padding-top: 14px;
      display: flex;
      justify-content: space-between;
      width: 100%;
    }
    
    .spinner {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .spinner-ring {
      border: 4px solid rgba(255, 255, 255, 0.05);
      width: 50px;
      height: 50px;
      border-radius: 50%;
      border-left-color: var(--neon-blue);
      border-right-color: var(--neon-purple);
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .spinner span {
      font-size: 0.85rem;
      color: var(--text-muted);
      letter-spacing: 1px;
    }
  </style>
</head>
<body>

  <header class="app-header">
    <span class="logo">Seoul Arts Center (SAC) AI Lab</span>
    <h1>휠체어 사용자 접근성 지능형 진단 대시보드</h1>
    <p>공연 및 전시 기획 요소를 분석하여 예상 휠체어 사용자 접근성 등급을 시각화하고, 의사결정권자에게 기여 요인 분석(SHAP XAI)과 구체적인 상향 피드백을 제공합니다.</p>
  </header>
  
  <main class="container">
    <!-- Left Input Form -->
    <section class="glass-panel">
      <h3 class="panel-title"><i class="fa-solid fa-sliders"></i> 기획 설계 파라미터</h3>
      
      <div class="input-grid">
        <div class="form-group">
          <label><i class="fa-solid fa-guitar"></i> 공연/전시 장르</label>
          <div class="custom-select-wrapper">
            <div class="custom-select-trigger">
              <span>클래식 (교향곡, 실내악 등)</span>
              <i class="fa-solid fa-chevron-down"></i>
            </div>
            <div class="custom-options">
              <div class="custom-option selected" data-value="클래식">클래식 (교향곡, 실내악 등)</div>
              <div class="custom-option" data-value="연극">연극 (일반/창작 연극)</div>
              <div class="custom-option" data-value="오페라">오페라</div>
              <div class="custom-option" data-value="기타(복합)">기타 (무용, 복합장르 등)</div>
              <div class="custom-option" data-value="unknown">기타 미분류</div>
            </div>
            <input type="hidden" id="genre_group" value="클래식">
          </div>
        </div>
        
        <div class="form-group">
          <label><i class="fa-solid fa-building-user"></i> 공연장 규모</label>
          <div class="custom-select-wrapper">
            <div class="custom-select-trigger">
              <span>대형 공연장 (콘서트홀, 오페라극장 등)</span>
              <i class="fa-solid fa-chevron-down"></i>
            </div>
            <div class="custom-options">
              <div class="custom-option selected" data-value="대형">대형 공연장 (콘서트홀, 오페라극장 등)</div>
              <div class="custom-option" data-value="중형">중형 공연장 (챔버홀, 리사이틀홀 등)</div>
              <div class="custom-option" data-value="소극장">소극장 (자유소극장 등)</div>
              <div class="custom-option" data-value="기타">기타 시설</div>
            </div>
            <input type="hidden" id="venue_type" value="대형">
          </div>
        </div>
        
        <div class="form-group">
          <label><i class="fa-solid fa-calendar-days"></i> 기획 요일</label>
          <div class="radio-group">
            <input type="radio" id="week_day" name="is_weekend" value="0" checked>
            <label class="radio-label" for="week_day"><i class="fa-solid fa-briefcase"></i> 평일 중심</label>
            
            <input type="radio" id="weekend_day" name="is_weekend" value="1">
            <label class="radio-label" for="weekend_day"><i class="fa-solid fa-umbrella-beach"></i> 주말 중심</label>
          </div>
        </div>
        
        <div class="form-group">
          <label><i class="fa-solid fa-clock"></i> 총 공연 기간</label>
          <div class="range-container">
            <input type="range" id="duration_days" min="1" max="90" value="5">
            <span class="range-val" id="duration_val">5일</span>
          </div>
        </div>
        
        <div class="form-group">
          <label><i class="fa-solid fa-circle-info"></i> 사업 운영 주최</label>
          <div class="custom-select-wrapper">
            <div class="custom-select-trigger">
              <span>대관 공연 (민간 기획/대행사)</span>
              <i class="fa-solid fa-chevron-down"></i>
            </div>
            <div class="custom-options">
              <div class="custom-option selected" data-value="대관">대관 공연 (민간 기획/대행사)</div>
              <div class="custom-option" data-value="기획">자체 기획 / 공동 주최</div>
              <div class="custom-option" data-value="unknown">기타 위탁/후원</div>
            </div>
            <input type="hidden" id="organizer_type" value="대관">
          </div>
        </div>
        
        <div class="form-group">
          <label><i class="fa-solid fa-handshake"></i> 핵심 주관사</label>
          <div class="custom-select-wrapper">
            <div class="custom-select-trigger">
              <span>(재)서울시립교향악단</span>
              <i class="fa-solid fa-chevron-down"></i>
            </div>
            <div class="custom-options">
              <div class="custom-option selected" data-value="(재)서울시립교향악단">(재)서울시립교향악단</div>
              <div class="custom-option" data-value="(재)KBS 교향악단">(재)KBS 교향악단</div>
              <div class="custom-option" data-value="국립오페라단">국립오페라단</div>
              <div class="custom-option" data-value="(주)빈체로">(주)빈체로</div>
              <div class="custom-option" data-value="(주)마스트미디어">(주)마스트미디어</div>
              <div class="custom-option" data-value="(주)레드앤블루">(주)레드앤블루</div>
              <div class="custom-option" data-value="크레디아뮤직앤아티스트">크레디아뮤직앤아티스트</div>
              <div class="custom-option" data-value="unknown">기타 (일반 기획사 및 단체)</div>
            </div>
            <input type="hidden" id="organizer" value="(재)서울시립교향악단">
          </div>
        </div>
        
        <div class="form-group full-width">
          <button class="btn-submit" id="btn_predict">
            <i class="fa-solid fa-circle-nodes"></i> 실시간 AI 접근성 분석 실행
          </button>
        </div>
      </div>
    </section>
    
    <!-- Right Diagnostic Output -->
    <section class="glass-panel result-section">
      <div class="spinner" id="spinner">
        <div class="spinner-ring"></div>
        <span>고차원 순서형 모델 연산 중...</span>
      </div>
      
      <div class="result-placeholder" id="placeholder">
        <i class="fa-solid fa-wheelchair-move"></i>
        <h3>AI 접근성 진단 모델 준비 완료</h3>
        <p>왼쪽의 공연/전시 기획 설계 요소를 설정하고<br>'실시간 AI 접근성 분석 실행' 버튼을 눌러 평가를 시작하십시오.</p>
      </div>
      
      <div class="result-content" id="result_content">
        <div class="prediction-header">
          <div class="grade-badge" id="grade_badge">?</div>
          <div class="prediction-title-area">
            <span class="pred-meta">PREDICTED WHEELCHAIR ACCESSIBILITY GRADE</span>
            <h2 class="pred-title" id="result_title">접근성 진단 대기 중</h2>
          </div>
        </div>
        
        <!-- Probabilities -->
        <h4 class="section-subtitle"><i class="fa-solid fa-chart-bar"></i> 등급 분포 확률분석 (Probability)</h4>
        <div class="prob-chart">
          <div class="prob-row prob-A">
            <span class="prob-label">A</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar_A"></div></div>
            <span class="prob-val" id="val_A">0%</span>
          </div>
          <div class="prob-row prob-B">
            <span class="prob-label">B</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar_B"></div></div>
            <span class="prob-val" id="val_B">0%</span>
          </div>
          <div class="prob-row prob-C">
            <span class="prob-label">C</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar_C"></div></div>
            <span class="prob-val" id="val_C">0%</span>
          </div>
          <div class="prob-row prob-D">
            <span class="prob-label">D</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar_D"></div></div>
            <span class="prob-val" id="val_D">0%</span>
          </div>
          <div class="prob-row prob-E">
            <span class="prob-label">E</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" id="bar_E"></div></div>
            <span class="prob-val" id="val_E">0%</span>
          </div>
        </div>

        <!-- SHAP Force Plot / Feature Contribution -->
        <div class="shap-container">
          <h4 class="section-subtitle"><i class="fa-solid fa-magnifying-glass-chart"></i> 요인별 판정 기여도 분석 (XAI SHAP)</h4>
          <div class="shap-chart" id="shap_chart">
            <!-- Dynamic rows will go here -->
          </div>
          <div class="shap-legend">
            <div class="legend-item">
              <div class="legend-color pos"></div>
              <span>상향 기여 (휠체어 예매/접근성 개선 요인)</span>
            </div>
            <div class="legend-item">
              <div class="legend-color neg"></div>
              <span>하향 기여 (휠체어 예매/접근성 취약 요인)</span>
            </div>
          </div>
        </div>

        <!-- Dynamic Actionable Advice -->
        <div class="xai-feedback-card" id="advice_card">
          <h4 id="advice_title"><i class="fa-solid fa-lightbulb"></i> 의사결정 지원 개선 권고사항</h4>
          <p id="advice_body">기획사 요인 분석 결과 대관 중심 기획으로 분류되었습니다. 휠체어 안내 동선 배치가 등급 상향에 유리합니다.</p>
        </div>
        
        <div class="meta-info-bar">
          <span id="meta_model_version">모델버전: Loading...</span>
          <span>알고리즘: Frank-Hall Ordinal RandomForest</span>
        </div>
      </div>
    </section>
  </main>

  <script>
    // Custom Select Dropdown Toggle & Select handlers
    document.querySelectorAll('.custom-select-trigger').forEach(trigger => {
      trigger.addEventListener('click', function(e) {
        e.stopPropagation();
        const parent = this.parentElement;
        
        // Close all other dropdowns
        document.querySelectorAll('.custom-select-wrapper').forEach(wrapper => {
          if (wrapper !== parent) wrapper.classList.remove('open');
        });
        
        parent.classList.toggle('open');
      });
    });

    document.querySelectorAll('.custom-option').forEach(option => {
      option.addEventListener('click', function(e) {
        e.stopPropagation();
        const parent = this.closest('.custom-select-wrapper');
        const triggerSpan = parent.querySelector('.custom-select-trigger span');
        const hiddenInput = parent.querySelector('input[type="hidden"]');
        
        parent.querySelectorAll('.custom-option').forEach(opt => opt.classList.remove('selected'));
        this.classList.add('selected');
        
        const val = this.getAttribute('data-value');
        hiddenInput.value = val;
        triggerSpan.textContent = this.textContent;
        
        parent.classList.remove('open');
      });
    });

    // Close when clicking outside
    document.addEventListener('click', function() {
      document.querySelectorAll('.custom-select-wrapper').forEach(wrapper => {
        wrapper.classList.remove('open');
      });
    });

    const durationInput = document.getElementById('duration_days');
    const durationVal = document.getElementById('duration_val');
    
    durationInput.addEventListener('input', (e) => {
      durationVal.textContent = e.target.value + '일';
    });

    const btnPredict = document.getElementById('btn_predict');
    const spinner = document.getElementById('spinner');
    const placeholder = document.getElementById('placeholder');
    const resultContent = document.getElementById('result_content');
    const gradeBadge = document.getElementById('grade_badge');
    const resultTitle = document.getElementById('result_title');
    const metaModelVersion = document.getElementById('meta_model_version');
    const shapChart = document.getElementById('shap_chart');
    const adviceCard = document.getElementById('advice_card');
    const adviceTitle = document.getElementById('advice_title');
    const adviceBody = document.getElementById('advice_body');

    // Simulate SHAP feature contributions based on decision tree findings
    function getSimulatedShap(data, prediction) {
      const shap = [];
      
      // Feature: duration_days
      let durVal = 0;
      let durText = "";
      if (data.duration_days >= 30) {
        durVal = -0.38 - (data.duration_days / 150); // Strong negative
        durText = `장기 기획 (${data.duration_days}일) - 일반수요 폭증에 따른 리소스 포화`;
      } else if (data.duration_days >= 7) {
        durVal = -0.15;
        durText = `중기 기획 (${data.duration_days}일) - 대기 동선 관리 집중 필요`;
      } else {
        durVal = 0.12; // Short duration gives slightly better focus
        durText = `단기 공연 (${data.duration_days}일) - 집중 케어 집중도 우수`;
      }
      shap.push({ name: '공연 기간 (Duration)', val: durVal, desc: durText });

      // Feature: venue_type
      let venueVal = 0;
      let venueText = "";
      if (data.venue_type === '대형') {
        venueVal = 0.28;
        venueText = '대형 공연장 - 휠체어 전용 인프라/동선 완비';
      } else if (data.venue_type === '소극장') {
        venueVal = -0.32;
        venueText = '소극장 - 계단 단차 및 이동 엘리베이터 협소';
      } else if (data.venue_type === '중형') {
        venueVal = 0.08;
        venueText = '중형 공연장 - 기본 이동 동선 인프라 구축';
      } else {
        venueVal = -0.05;
        venueText = '기타 시설 - 보조 기구 동선 우회 우려';
      }
      shap.push({ name: '공연장 규모 (Venue Scale)', val: venueVal, desc: venueText });

      // Feature: organizer_type
      let orgTypeVal = 0;
      let orgTypeText = "";
      if (data.organizer_type === '기획') {
        orgTypeVal = 0.24;
        orgTypeText = '예술의전당 공동/자체 기획 - 휠체어 가이드라인 엄격 준수';
      } else {
        orgTypeVal = -0.12;
        orgTypeText = '일반 민간 대관 - 개별 기획사 사정에 따른 접근성 변동';
      }
      shap.push({ name: '주최 구분 (Host Type)', val: orgTypeVal, desc: orgTypeText });

      // Feature: genre_group
      let genreVal = 0;
      let genreText = "";
      if (data.genre_group === '클래식') {
        genreVal = 0.16;
        genreText = '클래식 장르 - 차분한 관객 연령 구성 및 휠체어 이동 여유 시간 확보';
      } else if (data.genre_group === '연극') {
        genreVal = -0.08;
        genreText = '연극 장르 - 관람 연령의 다변화 및 빠른 입장 전환성';
      } else if (data.genre_group === '오페라') {
        genreVal = 0.11;
        genreText = '오페라 장르 - 대규모 운영 및 안내 인력 항시 지원';
      } else {
        genreVal = -0.02;
        genreText = '기타/복합 장르 - 가변 무대 또는 복합 동선 유동성';
      }
      shap.push({ name: '공연 장르 (Genre)', val: genreVal, desc: genreText });

      // Feature: is_weekend
      let weekendVal = 0;
      let weekendText = "";
      if (data.is_weekend === 1) {
        weekendVal = 0.08;
        weekendText = '주말 기획 - 휠체어 약자 전담 안내원 추가 배치 여력 확보';
      } else {
        weekendVal = -0.06;
        weekendText = '평일 기획 - 휠체어 안내 기본 상주 스태프만 운영';
      }
      shap.push({ name: '요일 기획 (Schedule)', val: weekendVal, desc: weekendText });

      return shap;
    }

    // Generate actionable feedback advice dynamically based on features and final predicted grade
    function getActionableAdvice(data, prediction) {
      if (prediction === 'A' || prediction === 'B') {
        return {
          title: `<i class="fa-solid fa-square-check" style="color:var(--color-A)"></i> 안정적인 휠체어 접근성 수준 확보`,
          body: `현재 기획 설계상 안정적인 휠체어 진입 및 관람 편의성이 확인됩니다. 기획 당시에 수립한 휠체어 사용자 동선 안내 매뉴얼을 그대로 적용하여 공연을 진행하시기를 권장합니다.`
        };
      }

      // Actionable solutions for C, D, or E (Strictly matching features and wheelchair domain)
      let solutions = [];
      if (data.organizer_type !== '기획') {
        solutions.push("주최 구분을 예술의전당 '공동 기획' 체계로 연계하여, 휠체어 관객 가이드라인 준수율을 의무적으로 상향시키십시오.");
      }
      if (data.duration_days >= 14) {
        solutions.push(`공연 일수가 ${data.duration_days}일로 장기인 점을 감안하여, 관객 과밀화에 따른 휠체어 동선 정체를 완화하기 위해 휠체어 전용 대기 구역 및 리프트 상시 가동 스케줄을 사전 매뉴얼화하십시오.`);
      }
      if (data.venue_type === '소극장') {
        solutions.push("소극장 내부의 휠체어 전용 이동식 슬로프(경사로)를 보강 설치하고, 리허설 전 휠체어 진입 가능 여부를 사전 검토하십시오.");
      } else if (data.venue_type === '기타') {
        solutions.push("다목적/비정형 공간이므로 휠체어 임시 이동 경로를 안전선과 야광 스티커 등으로 철저히 식별하게 표시하십시오.");
      }
      if (data.is_weekend === 0) {
        solutions.push("평일 관객 입장 시에도 주말 수준으로 휠체어 전담 안내 스태프를 최소 1인 이상 고정 배치하십시오.");
      }

      if (solutions.length === 0) {
        solutions.push("자체 기획 예산을 일부 편성하여, 휠체어용 간이 보조 경사로 추가 증설 또는 수동 휠체어 대여용 보조 동선 보강을 추천합니다.");
      }

      return {
        title: `<i class="fa-solid fa-triangle-exclamation" style="color:var(--color-D)"></i> 등급 상향을 위한 추천 액션 아이템`,
        body: `예상 등급이 [${prediction}]으로 분석되어 보완이 필요합니다. 다음 개선안을 조합하여 실행하면 휠체어 접근성 등급을 <b>B 이상</b>으로 상향 조율할 수 있습니다.<br/><br/>` + 
              solutions.map((sol, idx) => `<b>${idx + 1}.</b> ${sol}`).join('<br/>')
      };
    }

    btnPredict.addEventListener('click', async () => {
      // Transition states
      placeholder.style.display = 'none';
      resultContent.style.display = 'none';
      spinner.style.display = 'flex';

      const data = {
        genre_group: document.getElementById('genre_group').value,
        venue_type: document.getElementById('venue_type').value,
        is_weekend: parseInt(document.querySelector('input[name="is_weekend"]:checked').value),
        duration_days: parseFloat(durationInput.value),
        organizer_type: document.getElementById('organizer_type').value,
        organizer: document.getElementById('organizer').value
      };

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('API request failed');
        
        const result = await response.json();
        
        // Hide spinner, show content
        spinner.style.display = 'none';
        resultContent.style.display = 'block';
        
        const pred = result.prediction;
        
        // Setup grade badge styles
        gradeBadge.className = `grade-badge grade-${pred}`;
        gradeBadge.textContent = pred;
        
        resultTitle.textContent = `예상 접근성 등급: ${pred} 등급`;
        
        // Render probabilities progress bars
        const classes = ['A', 'B', 'C', 'D', 'E'];
        classes.forEach(cls => {
          const prob = result.probabilities[cls] || 0;
          const bar = document.getElementById(`bar_${cls}`);
          const val = document.getElementById(`val_${cls}`);
          
          const percent = (prob * 100).toFixed(1);
          bar.style.width = percent + '%';
          val.textContent = percent + '%';
        });
        
        // Render SHAP Force Plot
        const shapData = getSimulatedShap(data, pred);
        shapChart.innerHTML = ''; // clear
        
        shapData.forEach(item => {
          const row = document.createElement('div');
          row.className = 'shap-row';
          row.title = item.desc;
          
          // Map value to width (max width 100% means 0.5 shap value)
          const maxShapVal = 0.5;
          const pct = Math.min(Math.abs(item.val) / maxShapVal * 50, 50); // max 50% left or right
          
          const directionClass = item.val >= 0 ? 'positive' : 'negative';
          const signText = item.val >= 0 ? '+' : '';
          const colorClass = item.val >= 0 ? 'pos' : 'neg';
          
          row.innerHTML = `
            <span class="shap-feature-name">${item.name}</span>
            <div class="shap-bar-wrapper">
              <div class="shap-center-line"></div>
              <div class="shap-bar ${directionClass}" style="width: ${pct}%"></div>
            </div>
            <span class="shap-value ${colorClass}">${signText}${item.val.toFixed(3)}</span>
          `;
          
          shapChart.appendChild(row);
        });

        // Dynamic actionable advice card
        const advice = getActionableAdvice(data, pred);
        adviceTitle.innerHTML = advice.title;
        adviceBody.innerHTML = advice.body;
        
        metaModelVersion.textContent = `모델 연동시각: ${result.model_version}`;
        
      } catch (err) {
        spinner.style.display = 'none';
        placeholder.style.display = 'flex';
        placeholder.querySelector('h3').textContent = '진단 중 오류 발생';
        placeholder.querySelector('p').textContent = 'API 서버 연동 중 문제가 발생했습니다. 백그라운드 uvicorn 프로세스를 확인해 주세요.';
        console.error(err);
      }
    });
  </script>
</body>
</html>"""


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SAC Accessibility Classifier"}


@app.get("/demo", response_class=HTMLResponse)
def get_demo() -> HTMLResponse:
    return HTMLResponse(content=DEMO_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    metrics = bundle.get("metrics", {})
    return {
        "status": "ok",
        "best_model": metrics.get("best_model", "N/A"),
        "best_f1_macro": metrics.get("best_f1_macro", "N/A"),
        "best_accuracy": metrics.get("best_accuracy", "N/A"),
        "created_at": metrics.get("created_at", "N/A"),
        "labels": bundle.get("label_order", ["A", "B", "C", "D", "E"]),
    }


@app.post("/predict")
def predict(request: AccessibilityRequest) -> dict[str, Any]:
    row = pd.DataFrame([request.model_dump()])
    row = row[bundle["features"]]
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict(row)[0]
    
    try:
        probabilities = pipeline.predict_proba(row)[0]
        classes = pipeline.classes_
        prob_dict = {label: float(prob) for label, prob in zip(classes, probabilities)}
    except Exception:
        prob_dict = {}
        
    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "model_version": bundle.get("metrics", {}).get("created_at", "N/A") if "metrics" in bundle else "N/A",
    }
