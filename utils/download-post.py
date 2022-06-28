#!/usr/bin/env python

"""
Poor person's AF/EA forum/LW post downloader.
"""

################################################################################
# Dependencies
#   gql (pip install gql)
#   markdownify (pip install markdownify)
################################################################################

import re
import os
import sys
import argparse
import dateutil.parser
from gql import gql, Client
from markdownify import MarkdownConverter
from gql.transport.aiohttp import AIOHTTPTransport

#-------------------------------------------------------------------------------
# CLI args
#-------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
  'post_url',
  help = 'URL of an AF/LW/EA forum post'
)

parser.add_argument(
  '-f',
  '--format',
  default = 'markdown',
  help = 'Output format ("html" or "markdown")'
)

parser.add_argument(
  '--internal-converter',
  default = True,
  help = 'Whether to use the internal HTML to Markdown converter',
  action = argparse.BooleanOptionalAction
)

args = parser.parse_args()

post_url = args.post_url
output_format = args.format
use_internal_html_to_md_converter = args.internal_converter


#-------------------------------------------------------------------------------
# Request the post content
#-------------------------------------------------------------------------------

m = re.match('(https://[^/]*)/posts/([^/]*)', post_url)
website = m.groups()[0]
post_id = m.groups()[1]

service_url = f'{website}/graphql'
graphql_output_format = 'html' if use_internal_html_to_md_converter else output_format

query = gql(
'''
{
    post(
        input: {  
          selector: {
              _id: "''' + post_id + '''"
          }      
        }) 
    {
        result {
          title
          postedAt
          slug
          contents {
            ''' + graphql_output_format + '''
          }
        }
    }
}
''')

client = Client(transport = AIOHTTPTransport(url = service_url), fetch_schema_from_transport = True)
response = client.execute(query)['post']['result']


#-------------------------------------------------------------------------------
# Output a Jekyll ready post
#-------------------------------------------------------------------------------

def html_to_markdown(html):
  """Internal HMTL to Markdown converter"""
  class CustomConverter(MarkdownConverter):
      def convert_img(self, el, text, convert_as_inline):
        classes = ' '.join(el.parent.get('class'))
        result = ''
        result += '<figure class="classes">\n'
        result += f'{el}\n'
        result += '</figure>\n\n'
        return result

      def convert_span(self, el, text, convert_as_inline):
        classes = el.get('class')

        if classes:
          if 'footnote-reference' in classes:
            return f'[^{el.find_all("a")[0]["href"][len("#"):]}]'
          if 'mjx-char' in classes:
            return f'{el}'
          if 'mjx-math' in classes:
            return '\\\\(' + el.get("aria-label").replace("\\", "\\\\") + '\\\\)'

        lines = []
        for child in el.children:
          lines.append(super().process_tag(child, convert_as_inline))
        return ''.join(lines)

      def convert_ol(self, el, text, convert_as_inline):
        classes = el.get('class')

        if classes and 'footnotes' in classes:
          result = '---\n\n'
          for child in el.children:
            if 'footnote-item' in child['class']:
              id = child['id']
              content = child.find_all('div', {'class': 'footnote-content'})[0]
              result += f'[^{id}]: {super().process_tag(content, convert_as_inline)}\n\n'
          return result

        return super().convert_ol(el, text, convert_as_inline)

      def convert_style(self, el, text, convert_as_inline):
        return ''

      def convert_figure(self, el, text, convert_as_inline):
        img = el.find('img')
        if img:
          del img['srcset']
        return f'{el}\n\n'

      def convert_figcaption(self, el, text, convert_as_inline):
        return f'{el}\n\n'

  html = contents
  markdown = CustomConverter().convert(html)
  return markdown

title    = response['title']
contents = response['contents'][graphql_output_format]
date_str = response['postedAt']
slug     = response['slug']

date = dateutil.parser.isoparse(date_str)

if output_format == 'markdown' and graphql_output_format == 'html':
  contents = html_to_markdown(contents)

filename = f'{date.strftime("%Y-%m-%d")}-{slug}.{"md" if output_format == "markdown" else "html"}'
file_contents = f'''---
layout: post
title: "{title}"
date: {date.strftime('%Y-%m-%d %H:%M:%S %Z')}
#subtitle: 
#background:
---

{contents}
'''

self_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(self_path, 'output')
output_path = os.path.join(output_dir, filename)

os.makedirs(output_dir, exist_ok = True)
with open(output_path, 'w') as f:
  f.write(file_contents)

print('Done.')
print(f'Output file: {output_path}')
