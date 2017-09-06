from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.layouts import widgetbox,column,row
from bokeh.models.widgets import Select
from bokeh.io import curdoc, show
from bokeh.client import push_session
import pandas as pd



gotBattles1=pd.read_csv('battles.csv')
gotBattles=gotBattles1[['name','region','location','year','attacker_1','attacker_commander',
						'defender_1','defender_commander','attacker_size','defender_size','major_death']]

#print(df.head())

source=ColumnDataSource(data=dict(name=[],region=[],location=[],
								year=[],attacker_1=[],attacker_commander=[],
								defender_1=[],defender_commander=[],
								attacker_size=[],defender_size=[],major_death=[]))


colorMapper=CategoricalColorMapper(factors=[0.0,1.0],
									palette=['#81d4bd','#DF1C63'])

hover=HoverTool(tooltips=[('Battle','@name'),
							('Region','@region'),
							('Location','@location'),
							('Year','@year'),
							('Attacker','@attacker_1'),
							('Attacker Commander','@attacker_commander'),
							('Defender','@defender_1'),
							('Defender Commander','@defender_commander'),
							('Attacker size','@attacker_size'),
							('Defender size','@defender_size')]
				)

p=figure(x_axis_label='Attacker Size',y_axis_label='Defender Size',
		 tools='wheel_zoom,reset,box_zoom,pan',
		 plot_width=600,
		 plot_height=400,
		 title="Battles in Game of Thrones")


p.circle('attacker_size','defender_size',source=source,
	color=dict(field='major_death',transform=colorMapper),
	legend='Major death',
	size=9,
	alpha=0.8,
	hover_alpha=1)
p.add_tools(hover)

# Select Region Widget (Select)
opts=list(set(gotBattles['region']))
opts.insert(0,"All")
selectRegion=Select(title="Region", options=opts, value="All")


def selectBattle():
	selected=gotBattles
	
	if (selectRegion.value!="All"):
		r=selectRegion.value
		selected=selected[selected.region==r]
	return selected


def update():
	df=selectBattle()
	
	source.data=dict(name=df['name'],region=df['region'],location=df['location'],
			year=df['year'],attacker_1=df['attacker_1'],attacker_commander=df['attacker_commander'],
			defender_1=df['defender_1'],defender_commander=df['defender_commander'],
			attacker_size=df['attacker_size'],defender_size=df['defender_size'], major_death=df['major_death'])
	
update()

selectRegion.on_change('value',lambda attr,old,new:update())
layout=row(selectRegion,p)


curdoc().add_root(layout)
curdoc().title = "zeBattles"
