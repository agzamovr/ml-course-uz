from IPython.display import display_html

def display_horizontal(dfs, result):
    template = """<div padding-bottom: 20px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>
    </div>"""
    html_str = ''
    for df in dfs:
        html_str += df.to_html()
    html_str = html_str.replace('table', 'table style="display:inline"')
    html_str = template.format(html_str) + result.to_html()
    display_html(html_str, raw=True)


def display_stacked(dfs, result):
    template = """<div style="float: left; padding-right: 20px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>
    </div>"""
    html_str = ''
    for df in dfs:
        html_str += df.to_html()
    html_str = template.format(html_str) + result.to_html()
    display_html(html_str, raw=True)