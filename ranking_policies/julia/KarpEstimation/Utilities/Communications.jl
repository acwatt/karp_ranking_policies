# Functions to send emails and texts with updates for long-running scripts

using SMTPClient
include("passwords.jl")  # google_app_pwd


function send_txt(subject, message; cc_email=true, verbose=false)
    opt = SendOptions(
    isSSL = true,
    username = "kratzer.canby@gmail.com",
    passwd = google_app_pwd,
    verbose=verbose
    )

    url = "smtp://smtp.gmail.com:587"

    to = ["<5033279232@msg.fi.google.com>"]
    from = "Julia Automated Txt <$(opt.username)>"
    from = "$(opt.username)"
    replyto = "$(opt.username)"
    cc = ["<aaron@acwatt.net>"]

    body = get_body(to, from, subject, message; cc, replyto)
    rcpt = vcat(to, cc)
    resp = send(url, rcpt, from, body, opt)
end

# send_txt("test_subject", "test_body")