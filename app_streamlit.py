# -*- coding: utf-8 -*-
import os
import sys
import io
import json
import base64
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 系統頁面設定 (必須在最頂端)
st.set_page_config(
    page_title="SLOPEGUARD PRO // 邊坡智慧分級",
    page_icon="⛰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 讀取 Logo
logo_b64 = ""
for potential_path in ["LOGO.webp", "logo.webp", "LOGO.png", "logo.png"]:
    if os.path.exists(potential_path):
        try:
            with open(potential_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode('utf-8')
            break
        except Exception:
            pass

logo_img_html = f'<img src="data:image/webp;base64,{logo_b64}" style="height:18px; vertical-align:middle; margin-right:6px;" />' if logo_b64 else ""

# ==============================================================================
# 行動端 UI 深度優化 CSS (自動適應深/淺色模式)
# ==============================================================================
st.markdown("""
<style>
    /* 壓縮全域留白與隱藏無用元素 */
    .block-container { padding-top: 1.2rem !important; padding-bottom: 3.5rem !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* 緊湊型標題設計 (RWD)，使用 var(--text-color) 適應深淺色主題 */
    .app-title { font-size: 22px; font-weight: 900; margin-bottom: 4px; color: var(--text-color); line-height: 1.2; }
    .app-subtitle { font-size: 12px; color: gray; margin-bottom: 12px; letter-spacing: 0.5px; }
    
    /* 響應式 2x2 數據卡片網格 */
    .metrics-wrapper {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-bottom: 16px;
    }
    @media (min-width: 768px) {
        .app-title { font-size: 28px; }
        .app-subtitle { font-size: 14px; margin-bottom: 20px; }
        .metrics-wrapper { grid-template-columns: repeat(4, 1fr); gap: 12px; }
    }
    
    .metric-card {
        background-color: var(--secondary-background-color); 
        border: 1px solid var(--border-color);
        padding: 8px 10px; border-radius: 6px; border-left: 4px solid #00E5FF;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-title { color: var(--text-color); font-size: 11px; font-weight: bold; margin-bottom: 2px; opacity: 0.8; }
    .metric-val { color: var(--text-color); font-size: 16px; font-weight: 800; font-family: monospace; line-height: 1.2; }
    
    /* 底部固定署名欄 */
    .fixed-footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: var(--secondary-background-color); color: var(--text-color); text-align: center;
        padding: 6px 0; font-size: 11px; font-weight: bold;
        border-top: 1px solid var(--border-color); z-index: 99999;
        display: flex; justify-content: center; align-items: center;
    }
    
    /* 壓縮上傳區塊與 Tab 頁籤 */
    .stFileUploader > div { padding: 0 !important; }
    button[data-baseweb="tab"] { font-size: 13px !important; padding: 6px 12px !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 邊坡分析核心引擎
# ==============================================================================
class SlopeVisionEngine:
    @staticmethod
    def hex_to_bgr(hex_str):
        hex_str = hex_str.lstrip('#')
        return (int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16))

    @staticmethod
    def draw_styled_polygon(img, pts, color_bgr=(0, 0, 255), thickness=4, line_style="dashed"):
        num_pts = len(pts)
        if line_style == "solid":
            cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)
            return

        pattern = [28, 8, 6, 8] if line_style == "centerline" else [5, 6] if line_style == "dotted" else [20, 12]

        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts]
            dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
            dist = (dx**2 + dy**2)**0.5
            if dist == 0: continue
            vx, vy = dx / dist, dy / dist

            curr, pat_idx = 0.0, 0
            while curr < dist:
                seg_len = pattern[pat_idx % len(pattern)]
                is_draw = (pat_idx % 2 == 0)
                next_d = min(curr + seg_len, dist)

                if is_draw:
                    sp = (int(p1[0] + vx * curr), int(p1[1] + vy * curr))
                    ep = (int(p1[0] + vx * next_d), int(p1[1] + vy * next_d))
                    cv2.line(img, sp, ep, color_bgr, thickness, cv2.LINE_AA)

                curr = next_d
                pat_idx += 1

    @classmethod
    def analyze(cls, cv_img, sensitivity, min_area, suppress_road, road_dilate, y_range, x_range):
        h, w = cv_img.shape[:2]
        total_pixels = h * w
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        b, g, r = cv_img[:, :, 0].astype(np.float32), cv_img[:, :, 1].astype(np.float32), cv_img[:, :, 2].astype(np.float32)
        nexg = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-5)
        
        forest_mask = np.where((nexg > 0.015) | ((h_chan >= 20) & (h_chan <= 95) & (s_chan >= 20)), 255, 0).astype(np.uint8)
        k_forest = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, k_forest)
        veg_coverage = (cv2.countNonZero(forest_mask) / total_pixels) * 100.0

        mask_soil = cv2.inRange(hsv, np.array([3, 15, 30]), np.array([32, 255, 250]))
        mask_structure = cv2.inRange(v_chan, int(max(70, 110 - sensitivity * 0.3)), 255)
        color_candidates = cv2.bitwise_or(mask_soil, mask_structure)

        edges = cv2.Canny(gray, 40, 120)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_guided = cv2.dilate(edges, kernel_edge, iterations=1)

        hazard_raw = cv2.bitwise_and(color_candidates, edge_guided)
        hazard_filtered = cv2.bitwise_and(hazard_raw, cv2.bitwise_not(forest_mask))

        roi_filter = np.zeros((h, w), dtype=np.uint8)
        roi_filter[int(h * (y_range[0] / 100.0)):int(h * (y_range[1] / 100.0)), 
                   int(w * (x_range[0] / 100.0)):int(w * (x_range[1] / 100.0))] = 255
        hazard_filtered = cv2.bitwise_and(hazard_filtered, roi_filter)

        bridge_w, bridge_h = int(max(25, w * 0.012)), int(max(35, h * 0.02))
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, bridge_h))
        scar_closed = cv2.morphologyEx(hazard_filtered, cv2.MORPH_CLOSE, kernel_bridge)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        scar_clean = cv2.morphologyEx(scar_closed, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(scar_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspicious_polys, total_scar_area = [], 0
        min_pixels = total_pixels * (min_area / 100.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_pixels:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw > (w * 0.8) and bh < (h * 0.04): continue
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                poly = cv2.approxPolyDP(cnt, epsilon, True)
                suspicious_polys.append(poly.reshape(-1, 2))
                total_scar_area += area

        scar_ratio = (total_scar_area / total_pixels) * 100.0
        if scar_ratio > 4.0: status, color = "CRITICAL (崩滑)", "#FF5252"
        elif scar_ratio > 0.8: status, color = "WARNING (局部)", "#FFB300"
        else: status, color = "NORMAL (穩定)", "#00E676"

        metrics = {
            "w": w, "h": h, "res": f"{w} x {h}",
            "veg": f"{veg_coverage:.2f} %", "scar": f"{scar_ratio:.2f} %",
            "count": f"{len(suspicious_polys)} 處", "status": status, "color": color
        }
        return suspicious_polys, metrics

    @classmethod
    def render_polygons(cls, base_img, polys, color_hex="#FF0000", line_style="dashed", thickness=4):
        out = base_img.copy()
        color_bgr = cls.hex_to_bgr(color_hex)
        for pts in polys:
            cls.draw_styled_polygon(out, pts, color_bgr=color_bgr, thickness=thickness, line_style=line_style)
        return out


class SlopeReportGenerator:
    @staticmethod
    def build_excel(raw_img, annotated_bgr, metrics, polys, w_orig, h_orig, route_name, stake_mileage, slope_dir, orig_level, new_level, overview_desc, suggest_desc):
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            export_rows, zone_summaries = [], []
            for z_i, p_pts in enumerate(polys, 1):
                poly_area = cv2.contourArea(p_pts)
                zone_summaries.append({"單元": f"Zone {z_i}", "面積(px)": int(poly_area), "比例(%)": round((poly_area / (w_orig * h_orig)) * 100, 3)})
                for v_i, pt in enumerate(p_pts, 1): export_rows.append({"單元": f"Zone {z_i}", "序號": v_i, "X": int(pt[0]), "Y": int(pt[1])})

            pd.DataFrame(export_rows).to_excel(writer, index=False, sheet_name='節點坐標')
            pd.DataFrame(zone_summaries).to_excel(writer, index=False, sheet_name='單元彙整')
            ws = writer.book.create_sheet(title='分級表', index=0)
            ws['A1'] = f"{route_name} {stake_mileage} 邊坡滾動分級表"
        return output_excel.getvalue()

    @staticmethod
    def build_word(raw_img, annotated_bgr, metrics, polys, w_orig, h_orig, route_name, stake_mileage, slope_dir, orig_level, new_level, cost_val, overview_desc, suggest_desc, memo_desc=""):
        if not HAS_DOCX: return None
        doc = docx.Document()
        doc.add_paragraph(f"{route_name} {stake_mileage} 邊坡滾動分級表")
        out_docx = io.BytesIO()
        doc.save(out_docx)
        return out_docx.getvalue()

    @staticmethod
    def build_cad_dxf(polys, w_orig, h_orig, route_name, stake_mileage, line_style="dashed"):
        lines = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
        for z_i, poly in enumerate(polys, 1):
            lines.extend(["0", "POLYLINE", "8", "SLOPE", "66", "1", "70", "1"])
            for pt in poly: lines.extend(["0", "VERTEX", "8", "SLOPE", "10", f"{float(pt[0]):.3f}", "20", f"{float(h_orig - pt[1]):.3f}", "30", "0.0"])
            lines.extend(["0", "SEQEND"])
        lines.extend(["0", "ENDSEC", "0", "EOF"])
        return "\n".join(lines).encode('utf-8')


# ==============================================================================
# 側邊欄控制台 (設定項)
# ==============================================================================
with st.sidebar:
    st.header("🎛 邊坡控制台")
    st.subheader("輪廓標註樣式自訂")
    color_options = {"工程紅 (#FF0000)": "#FF0000", "青藍 (#00E5FF)": "#00E5FF", "警戒黃 (#FF