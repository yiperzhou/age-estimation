fgnet_raw_path = "/media/yi/harddrive/codes/Age-Gender-Pred/pics/FGNET"

def creat_fgnet_val(clean = False):
  if clean:
    clear_dir(config.val)
  pwd = os.getcwd()
  os.chdir(config.fgnet_raw)
  for pic in glob.glob("*"):
    name, age = re.findall(r'(\d)*A(\d*).*', pic)[0]
    newName = "{}_1_{}.jpg".format(age,
                                    name[0]
                                    .replace(' ', '')
                                    .replace('/', '')
                                    .replace(':', ''))
    # 2.1 check duplicate
    # 2.1 if duplicate exist, append a random number to it name
    newNameNoDupli = newName
    while os.path.exists(config.labeled + newNameNoDupli):
      newNameNoDupli = "{}{}{}".format(newName[:-4], random.randint(1, 9999), newName[-4:])
    # 2.2 save as a new file
    copyfile(config.fgnet_raw + pic, config.val + newNameNoDupli)
  os.chdir(pwd)
