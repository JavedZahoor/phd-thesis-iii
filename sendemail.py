#http://stackabuse.com/how-to-send-emails-with-gmail-using-python/
import smtplib
def SendEmail(sub,msg):
	#  174745924375-vq1r4m75psv1l7uk1leg291hgck0dbia.apps.googleusercontent.com 
	#  _P_XoX065wRv54WA9k-Q5FaG 
	gmail_user = 'phd.notifs@gmail.com'  
	gmail_password = 'CHmod512'

	from_ = gmail_user  
	to = ['javed.zahoor@gmail.com', 'sarim.zafar71@gmail.com']  
	subject = sub
	#body = "Hey, what's up?\n\n- You"

	email_text = """\  
	From: %s  
	To: %s  
	Subject: %s

	%s
	""" % (from_, ", ".join(to), subject, msg)

	try:  
		server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
		server.ehlo()
		server.login(gmail_user, gmail_password)
		server.sendmail(from_, to, email_text)
		server.close()

		print 'Email sent!'
	except:  
		print 'Something went wrong...'