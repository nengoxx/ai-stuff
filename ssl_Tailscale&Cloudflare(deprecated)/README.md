# Building an all in one solution to connect from anywhere with all the services(WIP, also there are better options built-in now)

	

The idea is to build a server side API endpoint with either Kobold.cpp or Oobabooga’s webui with SillyTavern as your UI to use remotely via voice input. We will need all the corresponding software and their dependencies on the computer you’re planning to use. In addition we will need several software to ensure a secure connection is available, there are several ways to achieve this and some might be better than others but I’m going to show the configuration I had before (which isn't a professional solution but works well for personal use).

We will assume you already set up the corresponding software for local use, launching the inference server with the ‘–api’ arguments and all the services (no need to use ‘–listen’ for all addresses, we will use the reverse proxy for accessing the server).

We will need to setup a reverse proxy in order to route the requests to the specific port for each service, such as the API endpoint of the Inference server, both for stable diffusion & oobaboga’s default port is 7860 so you’ll have to change one if you plan to use them in the same machine. For reference, these are the default ports for each software we’re gonna use:

[Oobabooga’s webui - 7860](http://localhost:7860)

[SillyTavern - 8000](http://localhost:8000)

[ST Extras (for speech recognition & other modules) - 5100](http://localhost:5100)

[Daswer123’s XTTS sever - 8020](http://localhost:8020)

[Automatic1111’s SD webui - 7860](http://localhost:7860)

[NginX proxy manager - 81](http://localhost:81)

First off you will need a way to connect to your server without the need to open any port on your router in order to prevent any kind of security issue, since this guide is just meant for personal use we’re not gonna publish any of the services to the internet. That’s done via a private VPN, there are many and many ways of achieving this, but for ease of use we will be setting up an already managed VPN with [Tailscale](https://tailscale.com/download) ([Zerotier](https://www.zerotier.com/download/) is also pretty similar).

There’s already documentation on how to install it and it's pretty simple so I’ll just say you need to install it in both your server and your client where you want to connect from, it has builds for linux, windows, mac and android, and if you need to install it on an unsupported device you can always [create a router to route the traffic thru a supported device](https://tailscale.com/kb/1019/subnets). 

Once you have the VPN up and running you can already connect remotely through your VPN’s IP address (something like 100.x.x.x) and the port you want to access, for connecting to your running ST instance for example, it will be something like ‘http://100.100.100.100:8000’ and you already have a way to use it remotely. The problem is as you might notice some functionality is not possible because the connection you are reaching (and what the server is providing) is insecure (no https), the speech recognition for example won't be routed thru the internet but rather use your local mic every time you try to use it. That happens despite using the VPN which already is secure because the services and your browser where you connect from have no way to know that.

For that problem we will use [Tailscale’s HTTPS solution](https://tailscale.com/kb/1153/enabling-https) and a piece of software called a [reverse proxy](https://www.cloudflare.com/learning/cdn/glossary/reverse-proxy/) ([NginX](https://www.nginx.com/) is among many other things a reverse proxy), which basically will translate the raw unencrypted data to securely access your services as if you were locally accessing them.

The first step is to follow [Tailscale documentation and enable HTTPS](https://tailscale.com/kb/1153/enabling-https), then create the certificates for each machine you want to connect to and save them with the command [tailscale cert](https://tailscale.com/blog/tls-certs). Once you have that setup we encounter a different problem, Tailscale provides a [very basic implementation of a DNS server](https://tailscale.com/kb/1081/magicdns#enabling-magicdns) that may or may not work for your setup, so we will disable it and use an external DNS provider, [Cloudflare](https://dash.cloudflare.com/login) in our case (you can also locally host a DNS server). To disable Tailscale MagicDNS [go to the admin dashboard > dns tab and disable magicDNS](https://login.tailscale.com/admin/dns), while we are here, add Cloudflare’s public DNS servers (1.1.1.1 & 1.0.0.1) and check ‘override local dns’. Also add cloudflare’s DNS servers that got assigned to your domain and add them as split dns for that domain.

![alt_text](cloudflareonts.png "image_tooltip")


The next step would be to add your domain/website in [Cloudflare’s dashboard](https://dash.cloudflare.com), the address should be the one Tailscale gave you when you enabled HTTPS (something like [dom-ain.ts.net](dom-ain.ts.net)), and add the DNS records pointing to the Tailscale address of your server (the 100.x.x.x IP), when adding the DNS records disable the proxying and just use the DNS service. You don’t need to add all your services, just point to the server and the reverse proxy will do the rest. Note that in order to claim the domain, you need to change your router's default DNS to the ones that cloudflare provided when you created the DNS records(at the bottom of the page), you can use the public DNS servers but [you’ll need to re-add the domain every month or so](https://developers.cloudflare.com/dns/zone-setups/troubleshooting/domain-deleted/).

If you can't change your router DNS for any reason you will have to set up a local DNS and connect to it from any machine you want to use remotely.

By now you should have the ability to connect to your computer from anywhere thru HTTPS and those DNS records as long as you have Tailscale running, but you need the actual proxy server to route to any subdomain or service. For that we are going to use [NginX proxy manager](https://nginxproxymanager.com/), it’s a dead easy implementation of NginX that runs on [Docker](https://www.docker.com/) (if you don’t want to run docker on your machine you can use NginX and configure it yourself). It’s a docker container so the IP address to access the local services outside docker will be [host.docker.internal](http://host.docker.internal).

The setup is very easy and the only thing you need to do is point to each of your services with the full [subdomain.dom-ain.ts.net](subdomain.domain.ts.net) address to the local IP of your server, which for docker will be for example: [http://host.docker.internal:8000](http://host.docker.internal:8000) for the SillyTavern UI. Remember to add the certificates you got from Tailscale in the SSL tab and connect through https://. The final configuration should look like this:

[sillytavern.dom-ain.ts.net](sillytavern.dom-ain.ts.net) -> [http://host.docker.internal:8000](http://host.docker.internal:8000)

[extras.dom-ain.ts.net](http://extras.dom-ain.ts.net) -> [http://host.docker.internal:5100](http://host.docker.internal:5100)

[xtts.dom-ain.ts.net](http://xtts.dom-ain.ts.net) -> [http://host.docker.internal:8020](http://host.docker.internal:8020)

[textgen.dom-ain.ts.net](http://textgen.dom-ain.ts.net) -> [http://host.docker.internal:7860](http://host.docker.internal:7860) (change one of the two if youre gonna use both)

[sd.dom-ain.ts.net](http://sd.dom-ain.ts.net) -> [http://host.docker.internal:7860](http://host.docker.internal:7860) (change one of the two if youre gonna use both)

Basically one subdomain for each service in order to use those addresses inside of SillyTavern or any UI that you want to use to connect from. 

Note that if you can't validate your domain in Cloudflare you’re gonna have to bypass the warnings from the browser you are connecting from, in firefox you just have to access every service once and click accept the risk and continue to the site.

At this point you can securely connect to SillyTavern from anywhere with all the services working as intended remotely.

