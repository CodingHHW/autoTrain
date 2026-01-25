#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修复labelImg的canvas.py文件中的类型转换问题
"""

import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_labelimg_canvas():
    """
    修复labelImg的canvas.py文件中的类型转换问题
    """
    try:
        # 获取labelImg的安装路径
        import labelImg
        labelimg_path = os.path.dirname(labelImg.__file__)
        logger.info(f"找到labelImg安装路径：{labelimg_path}")
        
        # 构建canvas.py的完整路径
        canvas_path = os.path.join(os.path.dirname(labelimg_path), "libs", "canvas.py")
        labelimg_py_path = os.path.join(labelimg_path, "labelImg.py")
        
        # 修复canvas.py文件
        if os.path.exists(canvas_path):
            # 读取文件内容
            with open(canvas_path, 'r') as f:
                canvas_content = f.read()
            
            # 查找需要修复的行
            canvas_fixes = [
                ("        p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())", 
                 "        p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())"),
                ("        p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())", 
                 "        p.drawLine(0, int(self.prev_point.y()), self.pixmap.width(), int(self.prev_point.y()))"),
                ("            p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)", 
                 "            p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))")
            ]
            
            # 检查是否需要修复
            canvas_needs_fix = False
            for old_line, new_line in canvas_fixes:
                if old_line in canvas_content:
                    canvas_needs_fix = True
                    break
            
            if canvas_needs_fix:
                # 修复文件
                fixed_canvas_content = canvas_content
                for old_line, new_line in canvas_fixes:
                    fixed_canvas_content = fixed_canvas_content.replace(old_line, new_line)
                
                # 保存修复后的文件
                with open(canvas_path, 'w') as f:
                    f.write(fixed_canvas_content)
                
                logger.info(f"成功修复canvas.py文件：{canvas_path}")
            else:
                logger.info("canvas.py文件已经修复，无需再次修改")
        else:
            logger.warning(f"canvas.py文件不存在：{canvas_path}")
        
        # 修复labelImg.py文件
        if os.path.exists(labelimg_py_path):
            # 读取文件内容
            with open(labelimg_py_path, 'r') as f:
                labelimg_content = f.read()
            
            # 查找需要修复的行
            labelimg_fixes = [
                ("    bar.setValue(bar.value() + bar.singleStep() * units)", 
                 "    bar.setValue(int(bar.value() + bar.singleStep() * units))")
            ]
            
            # 检查是否需要修复
            labelimg_needs_fix = False
            for old_line, new_line in labelimg_fixes:
                if old_line in labelimg_content:
                    labelimg_needs_fix = True
                    break
            
            if labelimg_needs_fix:
                # 修复文件
                fixed_labelimg_content = labelimg_content
                for old_line, new_line in labelimg_fixes:
                    fixed_labelimg_content = fixed_labelimg_content.replace(old_line, new_line)
                
                # 保存修复后的文件
                with open(labelimg_py_path, 'w') as f:
                    f.write(fixed_labelimg_content)
                
                logger.info(f"成功修复labelImg.py文件：{labelimg_py_path}")
            else:
                logger.info("labelImg.py文件已经修复，无需再次修改")
        else:
            logger.warning(f"labelImg.py文件不存在：{labelimg_py_path}")
        
        return True
            
    except PermissionError:
        logger.error("没有权限修改文件，请以管理员身份运行此脚本")
        return False
    except Exception as e:
        logger.error(f"修复labelImg文件时出错：{e}")
        return False


if __name__ == "__main__":
    fix_labelimg_canvas()