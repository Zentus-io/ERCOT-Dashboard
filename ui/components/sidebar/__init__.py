"""
Sidebar Component
Zentus - ERCOT Battery Revenue Dashboard

This package provides the main sidebar rendering function for the dashboard.
Re-exports render_sidebar() from render.py for backward compatibility.
"""

from .render import render_sidebar

__all__ = ['render_sidebar']
