# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['SweetKiss.py','BayesianOptimization.py','qss.py', 'figure.py'],
             pathex=["C:\\Users\\Administrator.000\\PycharmProjects\\forpacking"],
             binaries=[],
             datas=[],
             hiddenimports=["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='SweetKiss',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon='favicon64.ico',
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
