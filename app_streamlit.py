# -*- coding: utf-8 -*-
"""
交通部公路局 公路邊坡智慧分級管理與破壞辨識系統 (SLOPEGUARD PRO)
規範依據：公路局邊坡水因素總體檢及分級管理作業原則
維護單位：公路局東區養護工程分局南澳工務段
核心技術：歸一化超綠指標 (NExG) 剛性林相濾除 + 崩積土石/格梁特徵自動聚合
"""

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

# 檢查 Word 模組支援
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

st.set_page_config(
    page_title="SLOPEGUARD PRO // 邊坡智慧分級管理系統",
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

logo_img_html = ""
if logo_b64:
    logo_img_html = f'<img src="data:image/webp;base64,{logo_b64}" style="height:22px; vertical-align:middle; margin-right:8px; display:inline-block;" />'

st.markdown("""
<style>
    footer {visibility: hidden;}
    .fixed-footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #11141A; color: #00E5FF; text-align: center;
        padding: 8px 0; font-size: 13px; font-weight: bold;
        letter-spacing: 1px; border-top: 1px solid #232A3B; z-index: 99999;
        display: flex; justify-content: center; align-items: center;
    }
    .metric-card {
        background-color: #151922; border: 1px solid #232A3B;
        padding: 10px 14px; border-radius: 4px; border-left: 4px solid #00E5FF;
    }
    .metric-title { color: #8F9CAE; font-size: 12px; font-weight: bold; }
    .metric-val { color: #FFFFFF; font-size: 20px; font-weight: 800; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. 光學植被約束與崩積露頭特徵擷取核心
# ==============================================================================
class SlopeVisionEngine:
    @staticmethod
    def hex_to_bgr(hex_str):
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (b, g, r)

    @classmethod
    def draw_styled_polygon(cls, img, pts, color_bgr=(0, 0, 255), thickness=4, line_style="dashed"):
        num_pts = len(pts)
        if line_style == "solid":
            cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)
            return

        if line_style == "centerline":
            pattern = [28, 8, 6, 8]
        elif line_style == "dotted":
            pattern = [5, 6]
        else:
            pattern = [20, 12]

        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts]
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            dist = (dx**2 + dy**2)**0.5
            if dist == 0:
                continue
            vx, vy = dx / dist, dy / dist

            curr = 0.0
            pat_idx = 0
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
        """
        以物理植被指數約束林冠，主動偵測崩積砂土、裸露母岩及RC格子梁
        """
        h, w = cv_img.shape[:2]
        total_pixels = h * w

        # -------------------------------------------------------------
        # 步驟一：歸一化超綠指標 (NExG) 剛性排除完整林冠 (含葉隙深影)
        # -------------------------------------------------------------
        b = cv_img[:, :, 0].astype(np.float32)
        g = cv_img[:, :, 1].astype(np.float32)
        r = cv_img[:, :, 2].astype(np.float32)

        denom = 2.0 * g + r + b + 1e-5
        nexg = (2.0 * g - r - b) / denom

        # HSV 色彩空間輔助
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # 敏感度微調閾值 (越靈敏，越嚴格判定植生邊界)
        nexg_threshold = 0.04 - ((sensitivity - 50) * 0.001)
        canopy_nexg = nexg > nexg_threshold
        canopy_hsv = (h_chan >= 25) & (h_chan <= 95) & (s_chan >= 20)

        # 完整健康林冠遮罩
        forest_mask = np.where(canopy_nexg | canopy_hsv, 255, 0).astype(np.uint8)
        
        # 形態學閉合消除樹冠高頻細小葉隙，確保完整樹叢不碎裂
        k_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, k_leaf)
        veg_coverage = (cv2.countNonZero(forest_mask) / total_pixels) * 100.0

        # -------------------------------------------------------------
        # 步驟二：崩積砂土、風化紅土、母岩及混凝土格梁特徵主動提取
        # -------------------------------------------------------------
        # 1. 棕紅/黃褐崩積砂土 (Soil / Clay)
        mask_soil = cv2.inRange(hsv, np.array([5, 30, 40]), np.array([28, 255, 255]))
        
        # 2. 灰白混凝土格梁／露頭母岩 (Concrete Grid / Rock)
        mask_concrete = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 45, 255]))
        
        # 3. 合併潛在災害目標
        hazard_candidates = cv2.bitwise_or(mask_soil, mask_concrete)

        # 4. 嚴格扣除林冠區：只要屬於完整樹林，直接忽略不計
        scar_raw = cv2.bitwise_and(hazard_candidates, cv2.bitwise_not(forest_mask))

        # -------------------------------------------------------------
        # 步驟三：空間 ROI 高程與橫向限制
        # -------------------------------------------------------------
        roi_y_min = int(h * (y_range[0] / 100.0))
        roi_y_max = int(h * (y_range[1] / 100.0))
        roi_x_min = int(w * (x_range[0] / 100.0))
        roi_x_max = int(w * (x_range[1] / 100.0))
        roi_filter = np.zeros((h, w), dtype=np.uint8)
        roi_filter[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = 255
        scar_raw = cv2.bitwise_and(scar_raw, roi_filter)

        # -------------------------------------------------------------
        # 步驟四：道路邊界抑制 (僅作用於靠底端車道，絕不誤殺坡面)
        # -------------------------------------------------------------
        if suppress_road:
            road_zone = np.zeros((h, w), dtype=np.uint8)
            road_zone[int(h * 0.75):, :] = 255
            asphalt_mask = cv2.inRange(s_chan, 0, 28)
            asphalt_road = cv2.bitwise_and(asphalt_mask, road_zone)
            
            k_road = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.04), int(h * 0.01)))
            road_detected = cv2.morphologyEx(asphalt_road, cv2.MORPH_OPEN, k_road)
            if road_dilate > 0:
                k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
                road_detected = cv2.dilate(road_detected, k_dil, iterations=road_dilate)
            scar_raw = cv2.bitwise_and(scar_raw, cv2.bitwise_not(road_detected))

        # -------------------------------------------------------------
        # 步驟五：多尺度空間聚合 (連通整片格子梁破壞面)
        # -------------------------------------------------------------
        # 針對 5472 等超高解析度自動換算足夠跨越梁格的自適應核心
        bridge_w = int(max(15, w * 0.007))
        bridge_h = int(max(20, h * 0.012))
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, bridge_h))
        
        # 閉合連通格梁骨架與內嵌土石
        scar_connected = cv2.morphologyEx(scar_raw, cv2.MORPH_CLOSE, kernel_bridge)
        
        # 濾除散落微小噪點
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        scar_clean = cv2.morphologyEx(scar_connected, cv2.MORPH_OPEN, kernel_clean)

        # -------------------------------------------------------------
        # 步驟六：多邊形提取與幾何擬合
        # -------------------------------------------------------------
        contours, _ = cv2.findContours(scar_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspicious_polys = []
        total_scar_area = 0
        min_pixels = total_pixels * (min_area / 100.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_pixels:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # 排除純橫貫細帶 (如路緣標線)
                if bw > (w * 0.85) and bh < (h * 0.04):
                    continue

                # 幾何多邊形平滑貼合邊界
                epsilon = 0.0055 * cv2.arcLength(cnt, True)
                poly = cv2.approxPolyDP(cnt, epsilon, True)
                poly_pts = poly.reshape(-1, 2)
                total_scar_area += area
                suspicious_polys.append(poly_pts)

        scar_ratio = (total_scar_area / total_pixels) * 100.0
        if scar_ratio > 3.0:
            status, color = "CRITICAL (顯著裸露/格梁破壞/沖刷擴散)", "#FF5252"
        elif scar_ratio > 0.8:
            status, color = "WARNING (局部岩體裸露/沖刷)", "#FFB300"
        else:
            status, color = "NORMAL (邊坡穩定/植生完整)", "#00E676"

        metrics = {
            "w": w, "h": h, "res": str(w) + " x " + str(h),
            "veg": "{:.2f} %".format(veg_coverage),
            "scar": "{:.2f} %".format(scar_ratio),
            "count": str(len(suspicious_polys)) + " 處",
            "status": status, "color": color
        }
        return suspicious_polys, metrics

    @classmethod
    def render_polygons(cls, base_img, polys, color_hex="#FF0000", line_style="dashed", thickness=4):
        out = base_img.copy()
        color_bgr = cls.hex_to_bgr(color_hex)
        for pts in polys:
            cls.draw_styled_polygon(out, pts, color_bgr=color_bgr, thickness=thickness, line_style=line_style)
        return out


# ==============================================================================
# 2. 多格式匯出引擎 (Excel / Word / CAD DXF)
# ==============================================================================
class SlopeReportGenerator:
    @staticmethod
    def build_excel(raw_img, annotated_bgr, metrics, polys, w_orig, h_orig,
                    route_name, stake_mileage, slope_dir, orig_level, new_level,
                    overview_desc, suggest_desc):
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            export_rows = []
            zone_summaries = []
            for z_i, p_pts in enumerate(polys, 1):
                poly_area = cv2.contourArea(p_pts)
                zone_summaries.append({
                    "邊坡破壞單元編號": f"Zone {z_i}",
                    "頂點數量": len(p_pts),
                    "像素面積 (sq.px)": int(poly_area),
                    "佔坡面面積比例 (%)": round((poly_area / (w_orig * h_orig)) * 100, 3)
                })
                for v_i, pt in enumerate(p_pts, 1):
                    export_rows.append({
                        "邊坡破壞單元編號": f"Zone {z_i}",
                        "頂點序號": v_i,
                        "像素坐標_X": int(pt[0]),
                        "像素坐標_Y": int(pt[1]),
                        "原圖寬度": w_orig,
                        "原圖高度": h_orig
                    })

            df_coords = pd.DataFrame(export_rows)
            df_zones = pd.DataFrame(zone_summaries)
            df_coords.to_excel(writer, index=False, sheet_name='邊坡破壞節點坐標')
            df_zones.to_excel(writer, index=False, sheet_name='各單元面積彙整')

            wb = writer.book
            ws = wb.create_sheet(title='邊坡滾動分級表', index=0)

            font_title = Font(name="標楷體", size=16, bold=True)
            font_sub = Font(name="標楷體", size=14, bold=True)
            font_header = Font(name="標楷體", size=11, bold=True)
            font_content = Font(name="標楷體", size=11)
            font_note = Font(name="標楷體", size=10, italic=True)

            fill_header = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")
            thin = Side(border_style="thin", color="000000")
            border_cell = Border(left=thin, right=thin, top=thin, bottom=thin)

            col_widths = {'A': 16, 'B': 16, 'C': 16, 'D': 16, 'E': 16, 'F': 16, 'G': 16, 'H': 16}
            for col, w in col_widths.items():
                ws.column_dimensions[col].width = w

            ws.merge_cells('A1:H1')
            ws['A1'] = "交通部公路局東區養護工程分局"
            ws['A1'].font = font_title
            ws['A1'].alignment = Alignment(horizontal="center", vertical="center")

            ws.merge_cells('A2:H2')
            ws['A2'] = "邊坡滾動分級表"
            ws['A2'].font = font_sub
            ws['A2'].alignment = Alignment(horizontal="center", vertical="center")

            ws['A3'] = "養護單位"
            ws.merge_cells('B3:D3')
            ws['B3'] = "交通部公路局東區養護工程分局南澳工務段"
            ws['E3'] = "邊坡原有分級"
            ws.merge_cells('F3:H3')
            ws['F3'] = orig_level

            ws['A4'] = "邊坡位置"
            ws.merge_cells('B4:D4')
            ws['B4'] = f"{route_name} ({slope_dir})"
            ws['E4'] = "里 程"
            ws.merge_cells('F4:H4')
            ws['F4'] = stake_mileage

            ws['A5'] = "災害所需經費"
            ws.merge_cells('B5:D5')
            ws['B5'] = "列入年度養護/復建工程辦理"
            ws['E5'] = "邊坡等級調整"
            ws.merge_cells('F5:H5')
            ws['F5'] = f"{orig_level}  →  {new_level}"

            ws['A6'] = "現況概述"
            ws.merge_cells('B6:H6')
            ws['B6'] = f"{overview_desc} (AI 自動辨識崩塌裸露比例約 {metrics['scar']}，破壞單元共 {len(polys)} 處)"

            ws['A7'] = "擬改善情形"
            ws.merge_cells('B7:H7')
            ws['B7'] = suggest_desc

            ws.merge_cells('A8:D8')
            ws['A8'] = "邊坡滾動前 (原始巡檢照片)"
            ws.merge_cells('E8:H8')
            ws['E8'] = "邊坡滾動後 (AI 破壞特徵標註成果圖)"

            for r in range(9, 23):
                ws.row_dimensions[r].height = 20

            ws['A23'] = "備註"
            ws.merge_cells('B23:H23')
            ws['B23'] = "本巡檢評估成果由南澳工務段邊坡影像破壞辨識系統產製，經無人機原生解析度驗證與合規檢核。"

            ws['A24'] = "工務段:"
            ws.merge_cells('B24:D24')
            ws['B24'] = "段長：____________"
            ws['E24'] = "主管單位:"
            ws.merge_cells('F24:H24')
            ws['F24'] = "科長/分局長：____________"

            ws.merge_cells('A25:H25')
            ws['A25'] = "註：本報告表係用於邊坡調整適用之，1式3份報分局核定。"
            ws['A25'].font = font_note
            ws['A25'].alignment = Alignment(horizontal="left", vertical="center")

            for r in range(3, 25):
                for col_idx in range(1, 9):
                    cell = ws.cell(row=r, column=col_idx)
                    cell.border = border_cell
                    if col_idx in [1, 5] and r in [3, 4, 5]:
                        cell.fill = fill_header
                        cell.font = font_header
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    elif col_idx == 1 and r in [6, 7, 23, 24]:
                        cell.fill = fill_header
                        cell.font = font_header
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    elif r in [3, 4, 5, 24] and col_idx in [5]:
                        cell.fill = fill_header
                        cell.font = font_header
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    elif r == 8:
                        cell.fill = fill_header
                        cell.font = font_header
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    else:
                        cell.font = font_content
                        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

            try:
                from openpyxl.drawing.image import Image as OpenpyxlImage
                aspect_ratio = float(h_orig) / float(w_orig)
                thumb_w = 540
                thumb_h = int(thumb_w * aspect_ratio)

                raw_thumb = cv2.resize(raw_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                res_thumb = cv2.resize(annotated_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

                raw_buf = io.BytesIO(cv2.imencode('.png', raw_thumb)[1].tobytes())
                res_buf = io.BytesIO(cv2.imencode('.png', res_thumb)[1].tobytes())

                img_raw_obj = OpenpyxlImage(raw_buf)
                img_res_obj = OpenpyxlImage(res_buf)

                img_raw_obj.width = 460
                img_raw_obj.height = int(460 * aspect_ratio)
                img_res_obj.width = 460
                img_res_obj.height = int(460 * aspect_ratio)

                ws.add_image(img_raw_obj, "A9")
                ws.add_image(img_res_obj, "E9")
            except Exception:
                pass

        return output_excel.getvalue()

    @staticmethod
    def build_word(raw_img, annotated_bgr, metrics, polys, w_orig, h_orig,
                   route_name, stake_mileage, slope_dir, orig_level, new_level,
                   cost_val, overview_desc, suggest_desc, memo_desc=""):
        if not HAS_DOCX:
            return None

        doc = docx.Document()
        section = doc.sections[0]
        section.top_margin = Inches(0.787)
        section.bottom_margin = Inches(0.787)
        section.left_margin = Inches(0.787)
        section.right_margin = Inches(0.787)

        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_title.paragraph_format.space_before = Pt(0)
        p_title.paragraph_format.space_after = Pt(2)
        p_title.paragraph_format.line_spacing = 1.15
        
        run_t1 = p_title.add_run("交通部公路局東區養護工程分局\n")
        run_t1.font.name = "標楷體"
        run_t1.font.size = Pt(16)
        run_t1.font.bold = True
        run_t1._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '標楷體')

        run_t2 = p_title.add_run("邊坡滾動分級表")
        run_t2.font.name = "標楷體"
        run_t2.font.size = Pt(16)
        run_t2.font.bold = True
        run_t2._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '標楷體')

        table = doc.add_table(rows=10, cols=8)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False

        col_w = [Inches(1.05), Inches(0.85), Inches(0.75), Inches(0.80),
                 Inches(1.05), Inches(0.80), Inches(0.80), Inches(0.80)]
        for row in table.rows:
            for i, w in enumerate(col_w):
                row.cells[i].width = w

        def set_cell_props(cell, bg_color=None, align_h=WD_ALIGN_PARAGRAPH.CENTER):
            tcPr = cell._tc.get_or_add_tcPr()
            vAlign = parse_xml(r'<w:vAlign %s w:val="center"/>' % nsdecls('w'))
            tcPr.append(vAlign)
            if bg_color:
                shd = parse_xml(r'<w:shd %s w:fill="%s"/>' % (nsdecls('w'), bg_color))
                tcPr.append(shd)
            borders = parse_xml(
                r'<w:tcBorders %s>'
                r'  <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                r'  <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                r'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                r'  <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                r'</w:tcBorders>' % nsdecls('w')
            )
            tcPr.append(borders)
            p = cell.paragraphs[0]
            p.alignment = align_h
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0

        def fill_cell(cell, text, bold=False, font_size=10, bg_color=None, align_h=WD_ALIGN_PARAGRAPH.CENTER):
            set_cell_props(cell, bg_color, align_h)
            p = cell.paragraphs[0]
            p.text = ""
            for line_idx, line in enumerate(text.split('\n')):
                if line_idx > 0:
                    p.add_run().add_break()
                run = p.add_run(line)
                run.font.name = "Times New Roman"
                run.font.size = Pt(font_size)
                run.font.bold = bold
                run.font.color.rgb = RGBColor(0, 0, 0)
                run._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '標楷體')

        fill_cell(table.cell(0, 0), "養護單位", bold=True, bg_color="EDEDED")
        c_m0 = table.cell(0, 1)
        c_m0.merge(table.cell(0, 3))
        fill_cell(c_m0, "交通部公路局東區養護工程分局南澳工務段", font_size=10, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(0, 4), "邊坡\n原有分級", bold=True, bg_color="EDEDED")
        c_m1 = table.cell(0, 5)
        c_m1.merge(table.cell(0, 7))
        box_a = "■" if "A" in orig_level else "□"
        box_b = "■" if "B" in orig_level else "□"
        box_c = "■" if "C" in orig_level else "□"
        box_d = "■" if "D" in orig_level else "□"
        box_none = "■" if "無" in orig_level else "□"
        orig_boxes = f"{box_a}A級邊坡  {box_b}B級邊坡\n{box_c}C級邊坡  {box_d}D級邊坡\n{box_none}無"
        fill_cell(c_m1, orig_boxes, font_size=9.5, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(1, 0), "邊坡位置", bold=True, bg_color="EDEDED")
        c_m2 = table.cell(1, 1)
        c_m2.merge(table.cell(1, 3))
        fill_cell(c_m2, f"{route_name} ({slope_dir})", font_size=10, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(1, 4), "里 程", bold=True, bg_color="EDEDED")
        c_m3 = table.cell(1, 5)
        c_m3.merge(table.cell(1, 7))
        box_shun = "■" if "順" in slope_dir else "□"
        box_ni = "■" if "逆" in slope_dir else "□"
        fill_cell(c_m3, f"{stake_mileage}    {box_shun}順樁    {box_ni}逆樁", font_size=10, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(2, 0), "災害\n所需經費", bold=True, bg_color="EDEDED")
        c_m4 = table.cell(2, 1)
        c_m4.merge(table.cell(2, 3))
        fill_cell(c_m4, f"{cost_val} 元", font_size=10, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(2, 4), "邊坡\n等級調整", bold=True, bg_color="EDEDED")
        c_m5 = table.cell(2, 5)
        c_m5.merge(table.cell(2, 7))
        fill_cell(c_m5, f"{orig_level}  →  {new_level}", font_size=11, bold=True, align_h=WD_ALIGN_PARAGRAPH.CENTER)

        fill_cell(table.cell(3, 0), "現況概述", bold=True, bg_color="EDEDED")
        c_m6 = table.cell(3, 1)
        c_m6.merge(table.cell(3, 7))
        overview_text = f"{overview_desc}\n(AI 判釋裸露崩塌比例約 {metrics['scar']}，異常單元 {len(polys)} 處)"
        fill_cell(c_m6, overview_text, font_size=9.5, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        fill_cell(table.cell(4, 0), "擬改善情形", bold=True, bg_color="EDEDED")
        c_m7 = table.cell(4, 1)
        c_m7.merge(table.cell(4, 7))
        fill_cell(c_m7, suggest_desc, font_size=9.5, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        c_m8 = table.cell(5, 0)
        c_m8.merge(table.cell(5, 3))
        fill_cell(c_m8, "邊坡滾動前 (原始巡檢照片)", bold=True, bg_color="EDEDED")

        c_m9 = table.cell(5, 4)
        c_m9.merge(table.cell(5, 7))
        fill_cell(c_m9, "邊坡滾動後 (AI 破壞特徵標註成果圖)", bold=True, bg_color="EDEDED")

        c_img_l = table.cell(6, 0)
        c_img_l.merge(table.cell(6, 3))
        set_cell_props(c_img_l, align_h=WD_ALIGN_PARAGRAPH.CENTER)

        c_img_r = table.cell(6, 4)
        c_img_r.merge(table.cell(6, 7))
        set_cell_props(c_img_r, align_h=WD_ALIGN_PARAGRAPH.CENTER)

        aspect_ratio = float(h_orig) / float(w_orig)
        thumb_w = 720
        thumb_h = int(thumb_w * aspect_ratio)

        raw_thumb = cv2.resize(raw_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        res_thumb = cv2.resize(annotated_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        raw_tmp = io.BytesIO(cv2.imencode('.png', raw_thumb)[1].tobytes())
        res_tmp = io.BytesIO(cv2.imencode('.png', res_thumb)[1].tobytes())

        p_l = c_img_l.paragraphs[0]
        p_l.text = ""
        p_l.add_run().add_picture(raw_tmp, width=Inches(3.2))

        p_r = c_img_r.paragraphs[0]
        p_r.text = ""
        p_r.add_run().add_picture(res_tmp, width=Inches(3.2))

        fill_cell(table.cell(7, 0), "備註", bold=True, bg_color="EDEDED")
        c_m10 = table.cell(7, 1)
        c_m10.merge(table.cell(7, 7))
        memo_content = memo_desc if memo_desc else "本巡檢評估成果由南澳工務段邊坡影像破壞辨識系統產製，經無人機原生解析度驗證與合規檢核。"
        fill_cell(c_m10, memo_content, font_size=9.5, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        c_s1 = table.cell(8, 0)
        c_s1.merge(table.cell(8, 2))
        fill_cell(c_s1, "工務段:  段長：____________", font_size=10, bold=True, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        c_s2 = table.cell(8, 3)
        c_s2.merge(table.cell(8, 7))
        fill_cell(c_s2, "主管單位:  科長/分局長：____________", font_size=10, bold=True, align_h=WD_ALIGN_PARAGRAPH.LEFT)

        p_bottom = doc.add_paragraph()
        p_bottom.paragraph_format.space_before = Pt(4)
        p_bottom.paragraph_format.space_after = Pt(0)
        run_bottom = p_bottom.add_run("註：本報告表係用於邊坡調整適用之，1式3份報分局核定。")
        run_bottom.font.name = "Times New Roman"
        run_bottom.font.size = Pt(9)
        run_bottom.font.italic = True
        run_bottom._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '標楷體')

        out_docx = io.BytesIO()
        doc.save(out_docx)
        return out_docx.getvalue()

    @staticmethod
    def build_cad_dxf(polys, w_orig, h_orig, route_name, stake_mileage, line_style="dashed"):
        cad_linetype = "DASHED"
        if line_style == "solid":
            cad_linetype = "CONTINUOUS"
        elif line_style == "centerline":
            cad_linetype = "CENTER"
        elif line_style == "dotted":
            cad_linetype = "DOT"

        lines = []
        lines.extend(["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC"])
        lines.extend([
            "0", "SECTION", "2", "TABLES",
            "0", "TABLE", "2", "LAYER", "70", "1",
            "0", "LAYER", "2", "SLOPE_DISASTER_BOUNDARY", "70", "0", "62", "1", "6", cad_linetype,
            "0", "ENDTAB",
            "0", "ENDSEC"
        ])
        lines.extend(["0", "SECTION", "2", "ENTITIES"])

        for z_i, poly in enumerate(polys, 1):
            lines.extend([
                "0", "POLYLINE",
                "8", "SLOPE_DISASTER_BOUNDARY",
                "62", "1",
                "6", cad_linetype,
                "66", "1",
                "70", "1"
            ])
            for pt in poly:
                px = float(pt[0])
                py = float(h_orig - pt[1])
                lines.extend([
                    "0", "VERTEX",
                    "8", "SLOPE_DISASTER_BOUNDARY",
                    "10", f"{px:.3f}",
                    "20", f"{py:.3f}",
                    "30", "0.0"
                ])
            lines.extend(["0", "SEQEND"])

        lines.extend(["0", "ENDSEC", "0", "EOF"])
        return "\n".join(lines).encode('utf-8')


# ==============================================================================
# 側邊欄控制台
# ==============================================================================
with st.sidebar:
    st.header("🎛 邊坡控制台")
    st.markdown("---")

    st.subheader("輪廓標註樣式自訂")
    color_options = {
        "工程警戒紅 (#FF0000)": "#FF0000",
        "青藍色 (#00E5FF)": "#00E5FF",
        "警戒工程黃 (#FFD600)": "#FFD600",
        "高對比綠色 (#00E676)": "#00E676",
        "工程白色 (#FFFFFF)": "#FFFFFF"
    }
    sel_color_name = st.selectbox("標註線條顏色", list(color_options.keys()), index=0)
    stroke_color = color_options[sel_color_name]
    custom_color = st.color_picker("或直接選取任意自訂顏色", stroke_color)
    active_stroke_color = custom_color if custom_color != stroke_color else stroke_color

    line_style_map = {
        "工程虛線 (Dashed)": "dashed",
        "工程實線 (Solid)": "solid",
        "工程中心線 (Centerline)": "centerline",
        "工程點線 (Dotted)": "dotted"
    }
    sel_style_name = st.selectbox("標註線條種類", list(line_style_map.keys()), index=0)
    active_line_style = line_style_map[sel_style_name]

    active_stroke_width = st.slider("標註線條粗細 (px)", 1, 12, 4)

    st.markdown("---")
    st.subheader("道路避開與抑制設定")
    suppress_road = st.checkbox("智慧避開底部道路與瀝青鋪面", value=False)
    road_dilate = st.slider("底部道路遮蔽強度", 1, 6, 2)

    st.markdown("---")
    st.subheader("AI 破壞辨識參數")
    sensitivity = st.slider("辨識敏感度 (Sensitivity %)", 10, 100, 50, 5)
    min_area = st.slider("雜訊過濾閥值 (Min Area %)", 0.05, 2.0, 0.2, 0.05)

    st.markdown("---")
    st.subheader("手動高程/橫向區間 (ROI)")
    manual_y = st.slider("垂直高程區間 (%)", 0, 100, (0, 100))
    manual_x = st.slider("水平橫向區間 (%)", 0, 100, (0, 100))


st.title("⛰️ SLOPEGUARD PRO // 公路邊坡智慧分級管理系統")
st.caption("交通部公路局「公路邊坡分級管理機制」 ｜ 邊坡滾動分級表與 CAD/Word/Excel 匯出版")

tab_cv, tab_audit = st.tabs(["邊坡影像破壞辨識與物件導向編輯", "115 年 A 級邊坡降等合規審查"])

with tab_cv:
    uploaded_file = st.file_uploader("匯入巡查或無人機空拍照片 (支援 JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig = raw_img.shape[:2]

        polys_detected, metrics = SlopeVisionEngine.analyze(
            raw_img, sensitivity, min_area, suppress_road, road_dilate, manual_y, manual_x
        )

        if "user_polys" not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
            st.session_state.user_polys = [p.copy() for p in polys_detected]
            st.session_state.last_uploaded_file = uploaded_file.name

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown('<div class="metric-card"><div class="metric-title">植生覆蓋比例</div><div class="metric-val">' + metrics["veg"] + '</div></div>', unsafe_allow_html=True)
        m2.markdown('<div class="metric-card" style="border-left-color: #FFB300;"><div class="metric-title">裸露崩塌比例</div><div class="metric-val">' + metrics["scar"] + '</div></div>', unsafe_allow_html=True)
        m3.markdown('<div class="metric-card" style="border-left-color: #FF5252;"><div class="metric-title">標註破壞單元</div><div class="metric-val">' + str(len(st.session_state.user_polys)) + ' 處</div></div>', unsafe_allow_html=True)
        m4.markdown('<div class="metric-card" style="border-left-color: ' + metrics["color"] + ';"><div class="metric-title">初判危害等級</div><div class="metric-val" style="font-size:16px; color:' + metrics["color"] + ';">' + metrics["status"] + '</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        annotated_bgr = SlopeVisionEngine.render_polygons(
            raw_img, st.session_state.user_polys,
            color_hex=active_stroke_color, line_style=active_line_style, thickness=active_stroke_width
        )
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**CAM 01 // 原始照片 (Raw Footage)** `[" + str(w_orig) + " x " + str(h_orig) + "]`")
            st.image(raw_rgb, use_container_width=True)

        with col_right:
            st.markdown("**AI INFERENCE // 邊坡標註結果 (" + sel_style_name.split(' ')[0] + ")** `[" + str(w_orig) + " x " + str(h_orig) + "]`")
            st.image(annotated_rgb, use_container_width=True)

        res_jpg = cv2.imencode('.jpg', annotated_bgr)[1].tobytes()
        st.download_button(
            label="💾 下載原生解析度標註成果圖檔",
            data=res_jpg,
            file_name="slope_annotated.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

        st.markdown("---")

        # 讀取獨立 HTML 檔案渲染編輯器
        st.subheader("🖱️ 物件導向圖面直接編輯器 (含回復上一步驟與手動新增)")
        st.caption("【操作指引】：\n"
                   "• **↩ 回復上一步驟**：點擊按鈕或按 `Ctrl + Z` 可即時復原誤刪或拉錯的節點！\n"
                   "• **➕ 手動新增區域**：點擊切換模式，於圖面點擊左鍵畫點，按 `Enter` 閉合完成。\n"
                   "• **🗑️ 刪除功能**：選取多邊形後按『刪除區域』；於節點上按『右鍵』可刪除單一節點。")

        target_display_w = 1280
        aspect_ratio = float(h_orig) / float(w_orig)
        target_display_h = int(target_display_w * aspect_ratio)

        resized_for_web = cv2.resize(raw_img, (target_display_w, target_display_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', resized_for_web, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        polys_js = []
        for p in st.session_state.user_polys:
            pts = [[float(pt[0]), float(pt[1])] for pt in p]
            polys_js.append(pts)

        # 讀取外部 HTML 模板，傳入顏色、線型與粗細參數
        html_template_path = "canvas_editor.html"
        if os.path.exists(html_template_path):
            with open(html_template_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            html_content = html_content.replace("__IMG_ORIG_W__", str(w_orig))
            html_content = html_content.replace("__IMG_ORIG_H__", str(h_orig))
            html_content = html_content.replace("__STROKE_COLOR__", active_stroke_color)
            html_content = html_content.replace("__LINE_STYLE__", active_line_style)
            html_content = html_content.replace("__STROKE_WIDTH__", str(active_stroke_width))
            html_content = html_content.replace("__IMG_B64__", img_b64)
            html_content = html_content.replace("__POLYS_DATA__", json.dumps(polys_js))
            components.html(html_content, height=730)
        else:
            st.error("找不到 canvas_editor.html 檔案，請確認已將該檔案放置於 C:\\SA 目錄中。")

        c_in1, c_in2 = st.columns([3, 1])
        with c_in1:
            json_input = st.text_input("貼上圖面最新頂點坐標資料 (JSON)", placeholder="點擊上方『📋 複製最新編輯成果』後貼於此處...", label_visibility="collapsed")
        with c_in2:
            if st.button("套用並更新圖檔與台帳", type="primary", use_container_width=True):
                if json_input:
                    try:
                        data = json.loads(json_input)
                        st.session_state.user_polys = [np.array(p, dtype=np.int32) for p in data]
                        st.success("已成功同步編輯成果（含新增區域與復原狀態）！")
                        st.rerun()
                    except Exception:
                        st.error("JSON 格式不符，請重新複製！")

        st.markdown("---")

        # 多格式匯出
        st.subheader("📑 邊坡成果多格式匯出 (Word / Excel / CAD)")
        st.caption("填寫下列基本公文資料，可依工程需要一鍵下載標準 Excel 報表、Word 審查報告或 AutoCAD 向量圖檔。")

        c_rep1, c_rep2, c_rep3, c_rep4 = st.columns(4)
        with c_rep1:
            in_route = st.text_input("路線名稱", value="台9丁線")
        with c_rep2:
            in_stake = st.text_input("里程樁號", value="2k+700")
        with c_rep3:
            in_dir = st.selectbox("邊坡方向", ["上邊坡", "下邊坡"])
        with c_rep4:
            in_orig_level = st.selectbox("原有邊坡等級", ["B級邊坡", "A級邊坡", "C級邊坡", "D級邊坡", "無"])

        c_rep5, c_rep6 = st.columns(2)
        with c_rep5:
            in_new_level = st.selectbox("調整後等級", ["A級邊坡", "B級邊坡", "C+級邊坡", "C級邊坡", "D級邊坡"])
        with c_rep6:
            in_cost = st.text_input("災害所需經費 (元)", value="列入年度合約工程辦理")

        in_overview = st.text_area("現況概述", value="上邊坡格子梁護坡受強降雨沖刷掏空，噴凝土與底層岩體滑移露頭顯著，有持續擴大危害行車安全之虞。")
        in_suggest = st.text_area("擬改善情形", value="擬辦理刷坡清理浮石，補強型框格梁、掛網錨筋灌漿並增設橫向洩水孔以減緩水壓。")

        c_btn1, c_btn2, c_btn3 = st.columns(3)

        with c_btn1:
            excel_bytes = SlopeReportGenerator.build_excel(
                raw_img=raw_img, annotated_bgr=annotated_bgr, metrics=metrics,
                polys=st.session_state.user_polys, w_orig=w_orig, h_orig=h_orig,
                route_name=in_route, stake_mileage=in_stake, slope_dir=in_dir,
                orig_level=in_orig_level, new_level=in_new_level,
                overview_desc=in_overview, suggest_desc=in_suggest
            )
            st.download_button(
                label="📊 匯出 Excel 檔 (.xlsx 邊坡滾動分級表)",
                data=excel_bytes,
                file_name=f"Slope_Report_{in_route}_{in_stake}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with c_btn2:
            word_bytes = SlopeReportGenerator.build_word(
                raw_img=raw_img, annotated_bgr=annotated_bgr, metrics=metrics,
                polys=st.session_state.user_polys, w_orig=w_orig, h_orig=h_orig,
                route_name=in_route, stake_mileage=in_stake, slope_dir=in_dir,
                orig_level=in_orig_level, new_level=in_new_level,
                cost_val=in_cost, overview_desc=in_overview, suggest_desc=in_suggest
            )
            if word_bytes:
                st.download_button(
                    label="📄 匯出 Word 檔 (.docx 公文報表格式)",
                    data=word_bytes,
                    file_name=f"Slope_Report_{in_route}_{in_stake}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            else:
                st.warning("請先安裝 python-docx 套件以啟用 Word 匯出。")

        with c_btn3:
            dxf_bytes = SlopeReportGenerator.build_cad_dxf(
                polys=st.session_state.user_polys, w_orig=w_orig, h_orig=h_orig,
                route_name=in_route, stake_mileage=in_stake, line_style=active_line_style
            )
            st.download_button(
                label="📐 匯出 CAD 向量檔 (.dxf / DWG 相容)",
                data=dxf_bytes,
                file_name=f"Slope_Vector_{in_route}_{in_stake}.dxf",
                mime="application/dxf",
                use_container_width=True
            )

with tab_audit:
    st.subheader("A 級邊坡降至 C+ 或 C 級審查表 (115年5月修訂)")
    st.info("規範依據：東分局養字第1150004213號函。復建完成之 A 級邊坡需在汛期後檢核至少 1 個汛期成效，並備妥完整佐證與專家會議審查核可。")

    c1, c2 = st.columns(2)
    with c1:
        f_end = st.checkbox("1. 提出時間已在年度汛期正式結束後", value=True)
        f_seasons = st.number_input("2. 邊坡完工後歷經之完整汛期數量 (法規至少 1 個)", min_value=0, max_value=10, value=1)
        f_cause = st.checkbox("3. 檢具「致災原因探討」分析報告", value=True)
        f_mitigation = st.checkbox("4. 檢具「削減措施與工程成效說明」", value=True)
    with c2:
        f_stab = st.checkbox("5. 檢具數值邊坡安定分析成果 (Safety Factor)", value=True)
        f_img = st.checkbox("6. 檢具航照/UAV 影像地貌判釋報告", value=True)
        f_photos = st.checkbox("7. 檢具現場完工現況照片", value=True)
        f_expert = st.checkbox("8. 已由分局邀集專家學者/技師研商會議審核通過", value=False)

    if st.button("執行 115 年合規性檢核", type="primary", use_container_width=True):
        issues = []
        if not f_end:
            issues.append("未符合時限：依 115 年規定，必須於『年度汛期正式結束後』方得啟動降等申請。")
        if f_seasons < 1:
            issues.append("觀測期不足：復建工程完成後，必須歷經至少『1 個完整汛期』成效驗證。")
        missing = []
        if not f_cause: missing.append("致災原因探討")
        if not f_mitigation: missing.append("削減措施說明")
        if not f_stab: missing.append("安定分析計算成果")
        if not f_img: missing.append("航照/UAV 影像判釋")
        if not f_photos: missing.append("現場完工現況照片")
        if missing:
            issues.append("佐證資料缺漏：缺少 [" + ", ".join(missing) + "]")
        if not f_expert:
            issues.append("程序未完成：尚未由分局邀集專家學者或執業技師召開研商會議審查核可。")

        if len(issues) == 0:
            st.success("【審核通過】：各項法定要件均合規，准予由 A 級調降為 C+ 級或 C 級邊坡！")
        else:
            st.error("【審核未通過 / 暫緩調降】：尚有下列項目未符合規範：")
            for item in issues:
                st.write("• " + str(item))

st.write("")
st.write("")
st.write("")
st.markdown(f'<div class="fixed-footer">{logo_img_html}Powered by 公路局東區養護工程分局南澳工務段</div>', unsafe_allow_html=True)