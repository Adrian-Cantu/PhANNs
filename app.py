import html
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory , Markup
#import requests as rqq
import subprocess

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__)) 
UPLOAD_FOLDER = ROOT_FOLDER + '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'faa', 'fasta', 'gif', 'fa'])
import urllib
#from markupsafe import Markup





#from run_di_model import ann_result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def unescape(s):
	s = s.replace("&lt;", "<")
	s = s.replace("&gt;", ">")
	s = s.replace("&#34;",'"')
	# this has to be last:
	s = s.replace("&amp;", "&")
	return s

@app.template_filter('sorttable')
def sorttable_filter(s):
	s= s.replace('table id=','table class="sortable" id=')
	return s

def return_html_table(filename):
        cmd = ["python" , "run_di_model.py" , app.config['UPLOAD_FOLDER'] + '/' + filename]
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
        out,err = p.communicate()
        print(err)

        return out


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/")
def main():
	return redirect(url_for('upload_file'))
	#return render_template('index.html', table_code= Markup(return_html_table('A45_phage__orfs_wo_parens2.fasta').decode('utf8')))

#@app.route("/table")
#def table():
#	return(print_table())


@app.route('/uploads/<filename>')
def uploaded_file(filename):
#	cmd = ["python" , "run_di_model.py" , app.config['UPLOAD_FOLDER'] + '/' + filename]
#	p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
#                            stderr=subprocess.PIPE,
#                            stdin=subprocess.PIPE)
#	out,err = p.communicate()
#	return out
#   print(return_html_table(filename).decode('utf8'))
    return render_template('index.html', table_code= Markup(return_html_table(filename).decode('utf8')))

@app.route('/result',methods=['POST'])
def print_result():
	net=ann_result(UPLOAD_FOLDER + '/' + request.data['miname'])
	table=net.print_table()
	#print(table)
	return(table,500,'head')

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, 'static'),
		'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=80)
#	app.run(host="0.0.0.0", port=80)
