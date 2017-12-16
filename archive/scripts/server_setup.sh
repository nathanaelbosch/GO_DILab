# Machine Setup

# This as root to get the account set up
adduser hikaru
gpasswd -a hikaru sudo
cp -r /root/.ssh /home/hikaru/
chown -R hikaru /home/hikaru/.ssh
# Mount the additional disk - skip if none present
sudo mkfs.ext4 /dev/vdc
sudo mount /dev/vdc ~/Projekte
sudo chown -R hikaru ~/Projekte

# Server Security:
sudo vim /etc/ssh/sshd_config           # Set PasswordAuthentication no
sudo ufw allow OpenSSH
sudo ufw enable

# From now on ssh directly into hikaru@whateveripitsusing

sudo apt install fish unzip python3-venv htop cifs-utils nfd-common
sudo apt update
sudo apt upgrade

# Fish setup - You can all skip that as well as my dotfiles part
curl -L https://get.oh-my.fish | fish
omf install agnoster
# Get dotfiles
# mkdir Projekte
# cd Projekte
# git clone https://github.com/nathbo/dotfiles.git
# cd dotfiles
# python3 save_and_deploy.py

# Get Projekt runninng:
# I personally have it in ~/Projekte/GO_DILab. Adjust this as needed
mkdir Projekte && cd ~/Projekte
git clone https://github.com/nathbo/GO_DILab.git
cd GO_DILab
python3 -m venv venv
# sed -i 's/\$(venv)/(venv)/g' venv/bin/activate.fish       # There was some weird bug

. venv/bin/activate
pip install -r requirements.txt
pip install --upgrade pip

# Get the data to the server! Change paths here as needed
# This is executed on your computer, not on the VM!
scp ~/Projekte/GO_DILab/data/full_file.txt.zip smallvm:Projekte/GO_DILab/data
scp ~/Projekte/GO_DILab/data/dgs.zip smallvm:Projekte/GO_DILab/data

# Back on the machine:
cd ~/Projekte/GO_DILab/data && unzip full_file.txt.zip && cd ..


# Test with some script