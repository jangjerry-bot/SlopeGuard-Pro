class SlopeVisionEngine:
    @staticmethod
    def hex_to_bgr(hex_str):
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (b, g, r)

    @staticmethod
    def draw_styled_polygon(img, pts, color_bgr=(0, 0, 255), thickness=4, line_style="dashed"):
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
        專業工程級邊坡災害辨識（升級防誤判版）：
        強制引入 Canny 結構邊緣梯度與色彩聯合約束，徹底濾除林相陰影碎斑。
        """
        h, w = cv_img.shape[:2]
        total_pixels = h * w

        # 1. 色彩空間與灰階轉換
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # 2. 嚴格的 NExG 植生與林冠防護罩 (保護整片健康森林與陰影)
        b = cv_img[:, :, 0].astype(np.float32)
        g = cv_img[:, :, 1].astype(np.float32)
        r = cv_img[:, :, 2].astype(np.float32)
        nexg = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-5)
        
        # 擴大林冠包絡面，將樹冠間的深色葉隙一併防護
        forest_mask = np.where((nexg > 0.015) | ((h_chan >= 20) & (h_chan <= 95) & (s_chan >= 20)), 255, 0).astype(np.uint8)
        k_forest = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, k_forest)
        veg_coverage = (cv2.countNonZero(forest_mask) / total_pixels) * 100.0

        # 3. 災害特徵捕捉 (土石崩積面 + 混凝土格子梁露頭)
        # 調整土壤色彩範圍，更精準對應紅褐色崩積砂土
        mask_soil = cv2.inRange(hsv, np.array([3, 15, 30]), np.array([32, 255, 250]))
        # 混凝土結構面 (中高亮度、低飽和)
        mask_structure = cv2.inRange(v_chan, int(max(70, 110 - sensitivity * 0.3)), 255)
        
        color_candidates = cv2.bitwise_or(mask_soil, mask_structure)

        # 4. 【核心優化】結構邊緣梯度約束 (Canny Edge Constraint)
        # 格子梁與崩塌露頭必然伴隨顯著的幾何邊界，隨機樹冠則無規則邊緣
        edges = cv2.Canny(gray, 40, 120)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_guided = cv2.dilate(edges, kernel_edge, iterations=1)

        # 災害候選區 = 色彩符合 且 (具備結構邊緣 或 屬於強烈崩積土石面)
        hazard_raw = cv2.bitwise_and(color_candidates, edge_guided)
        
        # 強制完全排除林冠區
        hazard_filtered = cv2.bitwise_and(hazard_raw, cv2.bitwise_not(forest_mask))

        # 5. 空間 ROI 高程與橫向範圍限制
        roi_y_min = int(h * (y_range[0] / 100.0))
        roi_y_max = int(h * (y_range[1] / 100.0))
        roi_x_min = int(w * (x_range[0] / 100.0))
        roi_x_max = int(w * (x_range[1] / 100.0))
        roi_filter = np.zeros((h, w), dtype=np.uint8)
        roi_filter[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = 255
        hazard_filtered = cv2.bitwise_and(hazard_filtered, roi_filter)

        # 6. 大尺度矩形閉合 (將格子梁內部的格格縫隙自動融合成一個完整的災害單元)
        bridge_w = int(max(25, w * 0.012))
        bridge_h = int(max(35, h * 0.02))
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, bridge_h))
        scar_closed = cv2.morphologyEx(hazard_filtered, cv2.MORPH_CLOSE, kernel_bridge)
        
        # 剔除殘餘微小雜訊點
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        scar_clean = cv2.morphologyEx(scar_closed, cv2.MORPH_OPEN, kernel_open)

        # 7. 多邊形輪廓萃取與面積篩選
        contours, _ = cv2.findContours(scar_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspicious_polys = []
        total_scar_area = 0
        min_pixels = total_pixels * (min_area / 100.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_pixels:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # 排除底部橫向跨幅過大的道路車道雜訊
                if bw > (w * 0.8) and bh < (h * 0.04):
                    continue

                epsilon = 0.005 * cv2.arcLength(cnt, True)
                poly = cv2.approxPolyDP(cnt, epsilon, True)
                poly_pts = poly.reshape(-1, 2)
                total_scar_area += area
                suspicious_polys.append(poly_pts)

        scar_ratio = (total_scar_area / total_pixels) * 100.0
        if scar_ratio > 4.0:
            status, color = "CRITICAL (顯著裸露/格梁破壞/崩滑擴散)", "#FF5252"
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